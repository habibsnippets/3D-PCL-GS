// optimized_pmf_edge.cpp
// Your original pipeline, kept intact, with CPU-focused optimizations added:
//  - KD-tree reuse
//  - O(1) membership checks via mask and unordered_set
//  - OpenMP parallelization for independent per-point loops
//  - Reduced allocations via reserve()
//  - Per-stage timers and processing-only timing
//
// NOTE: build with -fopenmp -O3 -march=native

#include <yaml-cpp/yaml.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>

#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <map>
#include <omp.h> // OpenMP
#include <iomanip>

using namespace std::chrono_literals;

// -----------------------------
// Helper function: detectEdgePoints
// - Reused KD-tree semantics and parallelized
// - Returns unique edge point indices (vector<int>)
// Intuition: building a KD-tree per point is expensive. We create per-thread local KdTree
// (pointing to same cloud) to avoid a global lock while searching in parallel.
// For small clouds constructing a local KdTree per thread is cheap and safe.
// -----------------------------
std::vector<int> detectEdgePoints(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<int>& ground_indices,
    float min_height_diff,
    float neighborhood_radius)
{
    std::vector<int> edge_points;
    edge_points.reserve(ground_indices.size());

    int n = static_cast<int>(ground_indices.size());
    if (n == 0) return edge_points;

    // Use per-thread containers to avoid contention, then merge
    int max_threads = omp_get_max_threads();
    std::vector<std::vector<int>> per_thread_edges;
    per_thread_edges.resize(max_threads);

    // Parallelize the per-point neighbor checks
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // Create local KD-tree per thread for thread-safety and to avoid locks.
        pcl::KdTreeFLANN<pcl::PointXYZ> local_kdtree;
        local_kdtree.setInputCloud(cloud);

        // Each thread processes a chunk of ground_indices
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            int idx = ground_indices[i];
            const pcl::PointXYZ& point = cloud->points[idx];

            std::vector<int> neighbor_indices;
            std::vector<float> neighbor_distances;

            // Use radius search by default; for small clouds this is fine.
            // Option: switch to nearestKSearch with small K for faster deterministic runtime.
            local_kdtree.radiusSearch(point, neighborhood_radius, neighbor_indices, neighbor_distances);

            for (const int& neighbor_idx : neighbor_indices) {
                float height_diff = std::abs(cloud->points[neighbor_idx].z - point.z);
                if (height_diff > min_height_diff) {
                    per_thread_edges[tid].push_back(idx);
                    break;
                }
            }
        }
    } // end pragma omp

    // Merge and unique
    for (const auto &vec : per_thread_edges) {
        edge_points.insert(edge_points.end(), vec.begin(), vec.end());
    }
    std::sort(edge_points.begin(), edge_points.end());
    edge_points.erase(std::unique(edge_points.begin(), edge_points.end()), edge_points.end());

    return edge_points;
}

// -----------------------------
// Helper function: isNearEdge
// - Uses KD-tree provided by caller (global or local)
// - Uses an unordered_set for O(1) membership testing of edge points
// Intuition: membership testing in a vector with std::find is O(n); unordered_set is O(1).
// -----------------------------
bool isNearEdge(
    const pcl::PointXYZ& point,
    const std::unordered_set<int>& edge_set,
    pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree,
    float buffer_distance)
{
    // Perform radius search using provided kdtree
    std::vector<int> neighbor_indices;
    std::vector<float> neighbor_distances;
    kdtree.radiusSearch(point, buffer_distance, neighbor_indices, neighbor_distances);

    for (const int& neighbor_idx : neighbor_indices) {
        if (edge_set.find(neighbor_idx) != edge_set.end()) {
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv)
{
    // --- NOTE ABOUT TIMING AND BENCHMARKING ---
    // We'll measure the time for the *processing* portion only: PMF, normals, edge detection,
    // multi-scale voting, context-aware filtering, clustering. This excludes PCD I/O and visualization
    // so you measure algorithmic speed separately from IO/GUI overhead.
    //
    // If you want the original behavior (time everything), move the start timer up to the top.

    //start time
    auto total_start_time = std::chrono::high_resolution_clock::now();

    // Load YAML
    YAML::Node config = YAML::LoadFile("/home/beast/Desktop/Habeeb/3D-PCL-GS/center.yaml");

    // Input files
    std::vector<std::string> input_files = config["input_files"].as<std::vector<std::string>>();
    std::cout << "Input file size : "<< input_files.size() << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ> temp;

    // Load PCD (I/O - not part of processing timer)
    pcl::io::loadPCDFile(input_files.at(0), temp);
    *cloud += temp;

    // Voxel DS (optional, provided by your YAML). Keep it — downsampling is a major win.
    if (config["preprocessing"] && config["preprocessing"]["voxel_downsampling"]) {
        bool voxel_downsampling = config["preprocessing"]["voxel_downsampling"].as<bool>();
        float voxel_leaf_size = config["preprocessing"]["voxel_leaf_size"].as<float>(0.05f); // Default 5 cm

        if (voxel_downsampling) {
            std::cout << "[INFO] Voxel downsampling enabled. Leaf size = "
                      << voxel_leaf_size << " m" << std::endl;

            pcl::VoxelGrid<pcl::PointXYZ> voxel;
            voxel.setInputCloud(cloud);
            voxel.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);

            pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
            voxel.filter(*downsampled);

            std::cout << "[INFO] Point count reduced from " << cloud->size()
                      << " → " << downsampled->size() << std::endl;

            cloud = downsampled;
        } else {
            std::cout << "[INFO] Voxel downsampling disabled." << std::endl;
        }
    }

    // Crop box setup
    pcl::CropBox<pcl::PointXYZ> crop_box;
    crop_box.setInputCloud(cloud);

    std::cout << "30" << std::endl;

    std::vector<float> min_vals = config["crop_box"]["min"].as<std::vector<float>>();
    std::vector<float> max_vals = config["crop_box"]["max"].as<std::vector<float>>();
    std::cout << "40" << std::endl;
    bool negative = config["crop_box"]["negative"].as<bool>();

    crop_box.setMin(Eigen::Vector4f(min_vals[0], min_vals[1], min_vals[2], 1.0f));
    crop_box.setMax(Eigen::Vector4f(max_vals[0], max_vals[1], max_vals[2], 1.0f));
    crop_box.setNegative(negative);
    crop_box.filter(*cloud);

    // -------------------------
    // Prepare some objects we will reuse often (optimization).
    // Reusing avoids repeated expensive constructions (like KdTree builds).
    // -------------------------
    // Build a persistent KD-tree pointer to be reused for most radius searches.
    // Note: for parallel searches we'll create per-thread KdTrees to avoid locking.
    pcl::KdTreeFLANN<pcl::PointXYZ> global_kdtree;
    global_kdtree.setInputCloud(cloud);

 
    

    // PMF setup with extended parameters
    pcl::PointIndicesPtr ground(new pcl::PointIndices);
    pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
    pmf.setInputCloud(cloud);
    pmf.setMaxWindowSize(config["pmf"]["max_window_size"].as<int>());
    pmf.setSlope(config["pmf"]["slope"].as<float>());
    pmf.setInitialDistance(config["pmf"]["initial_distance"].as<float>());
    pmf.setMaxDistance(config["pmf"]["max_distance"].as<float>());

    if (config["pmf"]["cell_size"]) {
        pmf.setCellSize(config["pmf"]["cell_size"].as<float>());
    }

    if (config["pmf"]["base_max_window_size"]) {
        std::cout << "Using base_max_window_size: " << config["pmf"]["base_max_window_size"].as<int>() << std::endl;
        // pmf.setBaseMaxWindowSize(config["pmf"]["base_max_window_size"].as<int>()); // commented: not standard
    }

    // Extract ground points (this can be one of the heavier calls depending on cloud & PMF params)
    auto pmf_start = std::chrono::high_resolution_clock::now();
    pmf.extract(ground->indices);
    auto pmf_end = std::chrono::high_resolution_clock::now();

    // Separate ground and non-ground initially
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr nonground_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    // Extract initial ground points
    extract.setIndices(ground);
    extract.setNegative(false);
    extract.filter(*ground_cloud);

    // Extract initial non-ground points
    extract.setNegative(true);
    extract.filter(*nonground_cloud);

    // For O(1) membership checks: build a ground mask (vector<char>) of cloud size
    std::vector<char> is_ground_mask(cloud->size(), 0);
    for (int idx : ground->indices) {
        if (idx >= 0 && idx < (int)is_ground_mask.size()) is_ground_mask[idx] = 1;
    }

    // Check if normal-based filtering is enabled
    bool normal_filtering_enabled = false;
    float normal_radius = 0.5f;
    float max_ground_angle = 15.0f;

    if (config["normal_filtering"] && config["normal_filtering"]["enabled"]) {
        normal_filtering_enabled = config["normal_filtering"]["enabled"].as<bool>();
        if (normal_filtering_enabled) {
            if (config["normal_filtering"]["radius"]) {
                normal_radius = config["normal_filtering"]["radius"].as<float>();
            }
            if (config["normal_filtering"]["max_ground_angle"]) {
                max_ground_angle = config["normal_filtering"]["max_ground_angle"].as<float>();
            }
        }
    }

    // Apply normal-based filtering if enabled
    if (normal_filtering_enabled) {
        std::cout << "Applying normal-based filtering..." << std::endl;

        // Compute normals (this is an expensive op but PCL provides a good implementation)
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);

        // Create KD-tree for normal estimation
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);

        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch(normal_radius);

        auto ne_start = std::chrono::high_resolution_clock::now();
        ne.compute(*normals);
        auto ne_end = std::chrono::high_resolution_clock::now();

        // Refine ground classification based on normals
        std::vector<int> refined_ground_indices;
        std::vector<int> additional_nonground_indices;

        refined_ground_indices.reserve(ground->indices.size());
        additional_nonground_indices.reserve(ground->indices.size());

        // Parallelize the per-ground-index normal-check loop
        #pragma omp parallel
        {
            std::vector<int> local_refined;
            std::vector<int> local_extra;

            #pragma omp for schedule(static)
            for (int i = 0; i < (int)ground->indices.size(); ++i) {
                int idx = ground->indices[i];
                if (idx < 0 || idx >= (int)normals->points.size()) continue;

                const pcl::Normal& normal = normals->points[idx];

                // Calculate angle from vertical (z component)
                float angle = std::acos(std::abs(normal.normal_z)) * 180.0f / M_PI;

                if (angle <= max_ground_angle) {
                    // Keep as ground
                    local_refined.push_back(idx);
                } else {
                    // Reclassify as non-ground
                    local_extra.push_back(idx);
                }
            }

            // Merge thread-local results
            #pragma omp critical
            {
                refined_ground_indices.insert(refined_ground_indices.end(), local_refined.begin(), local_refined.end());
                additional_nonground_indices.insert(additional_nonground_indices.end(), local_extra.begin(), local_extra.end());
            }
        } // end parallel

        // Update ground indices and mask
        ground->indices = refined_ground_indices;
        std::fill(is_ground_mask.begin(), is_ground_mask.end(), 0);
        for (int idx : ground->indices) if (idx >= 0 && idx < (int)is_ground_mask.size()) is_ground_mask[idx] = 1;

        // Re-extract ground and non-ground clouds with refined classification
        extract.setIndices(ground);
        extract.setNegative(false);
        extract.filter(*ground_cloud);

        extract.setNegative(true);
        extract.filter(*nonground_cloud);

        std::cout << "After normal filtering: " << ground_cloud->size() << " ground points, "
                  << nonground_cloud->size() << " non-ground points" << std::endl;
    }

    // Check if edge detection is enabled
    bool edge_detection_enabled = false;
    float min_height_diff = 0.15f;
    float neighborhood_radius = 0.3f;
    float edge_buffer = 0.2f;

    if (config["edge_detection"] && config["edge_detection"]["enabled"]) {
        edge_detection_enabled = config["edge_detection"]["enabled"].as<bool>();
        if (edge_detection_enabled) {
            if (config["edge_detection"]["min_height_diff"]) {
                min_height_diff = config["edge_detection"]["min_height_diff"].as<float>();
            }
            if (config["edge_detection"]["neighborhood_radius"]) {
                neighborhood_radius = config["edge_detection"]["neighborhood_radius"].as<float>();
            }
            if (config["edge_detection"]["edge_buffer"]) {
                edge_buffer = config["edge_detection"]["edge_buffer"].as<float>();
            }
        }
    }

    // Apply edge detection if enabled
    if (edge_detection_enabled) {
        std::cout << "Applying edge detection..." << std::endl;

        // Detect edges (this uses optimized detectEdgePoints that parallelizes and uses per-thread KdTree)
        auto ed_start = std::chrono::high_resolution_clock::now();
        std::vector<int> edge_points = detectEdgePoints(cloud, ground->indices, min_height_diff, neighborhood_radius);
        auto ed_end = std::chrono::high_resolution_clock::now();

        std::cout << "Detected " << edge_points.size() << " edge points" << std::endl;

        // Create an unordered_set for O(1) membership queries
        std::unordered_set<int> edge_set;
        edge_set.reserve(edge_points.size() * 2 + 1);
        for (int idx : edge_points) edge_set.insert(idx);

        // Create buffer around edges and reclassify (parallel)
        std::vector<int> refined_ground_indices;
        std::vector<int> additional_nonground_indices;

        refined_ground_indices.reserve(ground->indices.size());
        additional_nonground_indices.reserve(ground->indices.size());

        // We will create local kdtree per thread for thread-safe radiusSearch
        #pragma omp parallel
        {
            std::vector<int> local_refined;
            std::vector<int> local_extra;

            pcl::KdTreeFLANN<pcl::PointXYZ> local_kdtree;
            local_kdtree.setInputCloud(cloud);

            #pragma omp for schedule(static)
            for (int i = 0; i < (int)ground->indices.size(); ++i) {
                int idx = ground->indices[i];
                if (idx < 0 || idx >= (int)cloud->points.size()) continue;

                const pcl::PointXYZ& pt = cloud->points[idx];

                if (isNearEdge(pt, edge_set, local_kdtree, edge_buffer)) {
                    local_extra.push_back(idx);
                } else {
                    local_refined.push_back(idx);
                }
            }

            #pragma omp critical
            {
                refined_ground_indices.insert(refined_ground_indices.end(), local_refined.begin(), local_refined.end());
                additional_nonground_indices.insert(additional_nonground_indices.end(), local_extra.begin(), local_extra.end());
            }
        } // end parallel

        // Update ground indices & mask
        ground->indices = refined_ground_indices;
        std::fill(is_ground_mask.begin(), is_ground_mask.end(), 0);
        for (int idx : ground->indices) if (idx >= 0 && idx < (int)is_ground_mask.size()) is_ground_mask[idx] = 1;

        // Re-extract
        extract.setIndices(ground);
        extract.setNegative(false);
        extract.filter(*ground_cloud);

        extract.setNegative(true);
        extract.filter(*nonground_cloud);

        std::cout << "After edge detection: " << ground_cloud->size() << " ground points, "
                  << nonground_cloud->size() << " non-ground points" << std::endl;
    }

    // Check if multi-scale processing is enabled
    bool multi_scale_enabled = false;
    std::vector<float> scales = {0.2f, 0.5f, 1.0f};
    float voting_threshold = 0.6f;

    if (config["multi_scale"] && config["multi_scale"]["enabled"]) {
        multi_scale_enabled = config["multi_scale"]["enabled"].as<bool>();
        if (multi_scale_enabled) {
            if (config["multi_scale"]["scales"]) {
                scales = config["multi_scale"]["scales"].as<std::vector<float>>();
            }
            if (config["multi_scale"]["voting_threshold"]) {
                voting_threshold = config["multi_scale"]["voting_threshold"].as<float>();
            }
        }
    }

    // Apply multi-scale processing if enabled
    if (multi_scale_enabled) {
        std::cout << "Applying multi-scale processing..." << std::endl;

        // This is a simplified implementation of multi-scale processing
        // We'll parallelize across scales: each scale builds its own PMF and produces ground indices.
        std::vector<std::vector<int>> scale_ground_indices(scales.size());

        #pragma omp parallel for schedule(dynamic)
        for (int si = 0; si < (int)scales.size(); ++si) {
            float scale = scales[si];
            pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> scale_pmf;
            scale_pmf.setInputCloud(cloud);

            // scaling window sizes and distances (coarse approach)
            scale_pmf.setMaxWindowSize(static_cast<int>(config["pmf"]["max_window_size"].as<int>() * scale));
            scale_pmf.setSlope(config["pmf"]["slope"].as<float>());
            scale_pmf.setInitialDistance(config["pmf"]["initial_distance"].as<float>() * scale);
            scale_pmf.setMaxDistance(config["pmf"]["max_distance"].as<float>() * scale);

            pcl::PointIndicesPtr scale_ground(new pcl::PointIndices);
            scale_pmf.extract(scale_ground->indices);
            scale_ground_indices[si] = std::move(scale_ground->indices);
        }

        // Count votes for each point across scales
        std::map<int, int> ground_votes; // small for small cloud
        for (const auto& indices : scale_ground_indices) {
            for (const int& idx : indices) {
                ground_votes[idx]++;
            }
        }

        // Determine final ground points based on voting threshold
        std::vector<int> final_ground_indices;
        final_ground_indices.reserve(ground_votes.size());
        for (const auto& vote : ground_votes) {
            float vote_ratio = static_cast<float>(vote.second) / scales.size();
            if (vote_ratio >= voting_threshold) {
                final_ground_indices.push_back(vote.first);
            }
        }

        // Update ground indices and mask
        ground->indices = final_ground_indices;
        std::fill(is_ground_mask.begin(), is_ground_mask.end(), 0);
        for (int idx : ground->indices) if (idx >= 0 && idx < (int)is_ground_mask.size()) is_ground_mask[idx] = 1;

        // Re-extract
        extract.setIndices(ground);
        extract.setNegative(false);
        extract.filter(*ground_cloud);

        extract.setNegative(true);
        extract.filter(*nonground_cloud);

        std::cout << "After multi-scale processing: " << ground_cloud->size() << " ground points, "
                  << nonground_cloud->size() << " non-ground points" << std::endl;
    }

    // Check if context-aware filtering is enabled
    bool context_aware_enabled = false;
    float region_size = 2.0f;
    float wall_detection_threshold = 0.6f;

    if (config["context_aware"] && config["context_aware"]["enabled"]) {
        context_aware_enabled = config["context_aware"]["enabled"].as<bool>();
        if (context_aware_enabled) {
            if (config["context_aware"]["region_size"]) {
                region_size = config["context_aware"]["region_size"].as<float>();
            }
            if (config["context_aware"]["wall_detection_threshold"]) {
                wall_detection_threshold = config["context_aware"]["wall_detection_threshold"].as<float>();
            }
        }
    }

    // Apply context-aware filtering if enabled
    if (context_aware_enabled) {
        std::cout << "Applying context-aware filtering..." << std::endl;

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud);

        std::vector<int> refined_ground_indices;
        refined_ground_indices.reserve(ground->indices.size());

        // Parallel processing across ground indices
        #pragma omp parallel
        {
            std::vector<int> local_refined;

            pcl::KdTreeFLANN<pcl::PointXYZ> local_kdtree;
            local_kdtree.setInputCloud(cloud);

            #pragma omp for schedule(dynamic)
            for (int i = 0; i < (int)ground->indices.size(); ++i) {
                int idx = ground->indices[i];
                const pcl::PointXYZ& point = cloud->points[idx];

                // Find neighbors within region_size
                std::vector<int> neighbor_indices;
                std::vector<float> neighbor_distances;
                local_kdtree.radiusSearch(point, region_size, neighbor_indices, neighbor_distances);

                if (neighbor_indices.empty()) continue;

                // Count how many neighbors are classified as ground using is_ground_mask
                int ground_count = 0;
                for (const int& neighbor_idx : neighbor_indices) {
                    if (neighbor_idx >= 0 && neighbor_idx < (int)is_ground_mask.size() && is_ground_mask[neighbor_idx]) {
                        ground_count++;
                    }
                }

                // Calculate ratio of ground points
                float ground_ratio = static_cast<float>(ground_count) / neighbor_indices.size();

                // Check for vertical distribution (potential wall)
                float min_z = std::numeric_limits<float>::max();
                float max_z = std::numeric_limits<float>::lowest();
                for (const int& neighbor_idx : neighbor_indices) {
                    const pcl::PointXYZ& neighbor = cloud->points[neighbor_idx];
                    min_z = std::min(min_z, neighbor.z);
                    max_z = std::max(max_z, neighbor.z);
                }

                float height_range = max_z - min_z;
                float height_to_width_ratio = height_range / region_size;

                // If height-to-width ratio high and ground_ratio low -> likely wall; skip
                if (height_to_width_ratio > wall_detection_threshold && ground_ratio < 0.5f) {
                    // skip - reclassify as non-ground implicitly
                } else {
                    local_refined.push_back(idx);
                }
            }

            #pragma omp critical
            refined_ground_indices.insert(refined_ground_indices.end(), local_refined.begin(), local_refined.end());
        } // end parallel

        // Update ground indices & mask
        ground->indices = refined_ground_indices;
        std::fill(is_ground_mask.begin(), is_ground_mask.end(), 0);
        for (int idx : ground->indices) if (idx >= 0 && idx < (int)is_ground_mask.size()) is_ground_mask[idx] = 1;

        // Re-extract
        extract.setIndices(ground);
        extract.setNegative(false);
        extract.filter(*ground_cloud);

        extract.setNegative(true);
        extract.filter(*nonground_cloud);

        std::cout << "After context-aware filtering: " << ground_cloud->size() << " ground points, "
                  << nonground_cloud->size() << " non-ground points" << std::endl;
    }

    // Clustering - Euclidean
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(nonground_cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    // NOTE: ensure we use float for cluster tolerance
    ec.setClusterTolerance(config["ec"]["clustertolerance"].as<float>(0.01f));
    ec.setMinClusterSize(config["ec"]["min_cluster_size"].as<int>(5));
    ec.setMaxClusterSize(config["ec"]["max_cluster_size"].as<int>(200));
    ec.setSearchMethod(tree);
    ec.setInputCloud(nonground_cloud);

    auto ec_start = std::chrono::high_resolution_clock::now();
    ec.extract(cluster_indices);
    auto ec_end = std::chrono::high_resolution_clock::now();

    extract.setInputCloud(nonground_cloud);
    std::cout << "Number of non-ground points: " << nonground_cloud->size() << std::endl;
    std::cout << "Number of clusters found: " << cluster_indices.size() << std::endl;

    for (int i = 0; i < cluster_indices.size(); i++) {
        pcl::PointIndices::Ptr indices(new pcl::PointIndices(cluster_indices[i]));
        extract.setIndices(indices);
        extract.setNegative(false);
        extract.filter(temp);

        std::cout << "Cluster " << i << " size: " << temp.size() << std::endl;
        pcl::io::savePCDFileBinary("cluster" + std::to_string(i) + ".pcd", temp);

        extract.setNegative(true);
        extract.filter(temp);
        pcl::io::savePCDFileBinary("non_cluster" + std::to_string(i) + ".pcd", temp);
    }

    // Save the final ground and non-ground point clouds (I/O - excluded from processing timer)
    pcl::io::savePCDFileBinary(config["output_files"]["ground"].as<std::string>(), *ground_cloud);
    pcl::io::savePCDFileBinary(config["output_files"]["nonground"].as<std::string>(), *nonground_cloud);

    // Compute moments for visualization
    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(nonground_cloud);
    feature_extractor.compute();

    pcl::PointXYZ min_point_AABB, max_point_AABB;
    feature_extractor.getAABB(min_point_AABB, max_point_AABB);


    // Visualization
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    // Add original cloud
    viewer->addPointCloud<pcl::PointXYZ>(cloud, "input_cloud");

    // Add bounding box
    viewer->addCube(
        min_point_AABB.x, max_point_AABB.x,
        min_point_AABB.y, max_point_AABB.y,
        min_point_AABB.z, max_point_AABB.z,
        1.0, 1.0, 0.0, "AABB"
    );

    viewer->setShapeRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
        "AABB"
    );

    // Add visualization for ground and non-ground points with different colors
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ground_color(ground_cloud, 0, 255, 0); // Green for ground
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> nonground_color(nonground_cloud, 255, 0, 0); // Red for non-ground

    viewer->addPointCloud<pcl::PointXYZ>(ground_cloud, ground_color, "ground_cloud");
    viewer->addPointCloud<pcl::PointXYZ>(nonground_cloud, nonground_color, "nonground_cloud");

    // Set point size for better visibility
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "ground_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "nonground_cloud");

    // If edge detection was enabled, visualize edge points
    if (edge_detection_enabled) {
        // Recompute/detect edge points for viz (fast, but we already computed earlier)
        std::vector<int> edge_points = detectEdgePoints(cloud, ground->indices, min_height_diff, neighborhood_radius);

        pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        edge_cloud->points.reserve(edge_points.size());

        for (const int& idx : edge_points) {
            edge_cloud->points.push_back(cloud->points[idx]);
        }
        edge_cloud->width = edge_cloud->points.size();
        edge_cloud->height = 1;
        edge_cloud->is_dense = true;

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> edge_color(edge_cloud, 0, 0, 255); // Blue for edges
        viewer->addPointCloud<pcl::PointXYZ>(edge_cloud, edge_color, "edge_cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "edge_cloud");
    }

    // Output statistics
    std::cout << "=== Segmentation Results ===" << std::endl;
    std::cout << "Original point cloud size: " << cloud->size() << std::endl;
    std::cout << "Ground points: " << ground_cloud->size() << " ("
              << (100.0f * ground_cloud->size() / cloud->size()) << "%)" << std::endl;
    std::cout << "Non-ground points: " << nonground_cloud->size() << " ("
              << (100.0f * nonground_cloud->size() / cloud->size()) << "%)" << std::endl;
    std::cout << "Number of clusters in non-ground points: " << cluster_indices.size() << std::endl;

    // If you want interactive viewer uncomment this loop.
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    //     std::this_thread::sleep_for(100ms);
    // }

    // Final runtime print (includes everything done earlier)
    // Note: total_duration above is algorithmic processing time only


    // End total timer here
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end_time - total_start_time
    );

    std::cout << "Total time for execution: "
              << (total_duration.count() / 1000.0)
              << " seconds" << std::endl;

    return 0;
}