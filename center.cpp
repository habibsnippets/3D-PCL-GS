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
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
using namespace std::chrono_literals;

// Helper function to detect edges in point cloud
std::vector<int> detectEdgePoints(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<int>& ground_indices,
    float min_height_diff,
    float neighborhood_radius) 
{
    std::vector<int> edge_points;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    for (const int& idx : ground_indices) {
        const pcl::PointXYZ& point = cloud->points[idx];
        
        // Find neighbors within radius
        std::vector<int> neighbor_indices;
        std::vector<float> neighbor_distances;
        kdtree.radiusSearch(point, neighborhood_radius, neighbor_indices, neighbor_distances);
        
        // Check height differences
        for (const int& neighbor_idx : neighbor_indices) {
            float height_diff = std::abs(cloud->points[neighbor_idx].z - point.z);
            if (height_diff > min_height_diff) {
                edge_points.push_back(idx);
                break;
            }
        }
    }
    
    return edge_points;
}

// Helper function to check if a point is near an edge
bool isNearEdge(
    const pcl::PointXYZ& point,
    const std::vector<int>& edge_points,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    float buffer_distance) 
{
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    
    std::vector<int> neighbor_indices;
    std::vector<float> neighbor_distances;
    kdtree.radiusSearch(point, buffer_distance, neighbor_indices, neighbor_distances);
    
    for (const int& neighbor_idx : neighbor_indices) {
        if (std::find(edge_points.begin(), edge_points.end(), neighbor_idx) != edge_points.end()) {
            return true;
        }
    }
    
    return false;
}

int main(int argc, char** argv)
{
    // Load YAML
    YAML::Node config = YAML::LoadFile("/home/habeeb-chisti/GRemove/center.yaml");

    // Input files
    std::vector<std::string> input_files = config["input_files"].as<std::vector<std::string>>();
    std::cout << "Input file size : "<< input_files.size() << std::endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ> temp;

    pcl::io::loadPCDFile(input_files.at(0), temp);
    *cloud += temp;
    
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

    // Extract additional parameters with defaults if not specified
    // PMF setup with extended parameters
    pcl::PointIndicesPtr ground(new pcl::PointIndices);
    pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
    pmf.setInputCloud(cloud);
    pmf.setMaxWindowSize(config["pmf"]["max_window_size"].as<int>());
    pmf.setSlope(config["pmf"]["slope"].as<float>());
    pmf.setInitialDistance(config["pmf"]["initial_distance"].as<float>());
    pmf.setMaxDistance(config["pmf"]["max_distance"].as<float>());
    
    // Set additional PMF parameters if they exist in config
    if (config["pmf"]["cell_size"]) {
        pmf.setCellSize(config["pmf"]["cell_size"].as<float>());
    }
    
    if (config["pmf"]["base_max_window_size"]) {
        // This parameter doesn't exist in standard PCL PMF, but we're showing how you would add it
        // You would need to modify the PCL library or extend the class
        std::cout << "Using base_max_window_size: " << config["pmf"]["base_max_window_size"].as<int>() << std::endl;
        // pmf.setBaseMaxWindowSize(config["pmf"]["base_max_window_size"].as<int>());
    }
    
    // Extract ground points
    pmf.extract(ground->indices);

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
        // Apply normal-based filtering if enabled
    if (normal_filtering_enabled) {
        std::cout << "Applying normal-based filtering..." << std::endl;
        
        // Compute normals
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        
        // Create KD-tree for normal estimation
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree);
        
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch(normal_radius);
        ne.compute(*normals);
        
        // Refine ground classification based on normals
        std::vector<int> refined_ground_indices;
        std::vector<int> additional_nonground_indices;
        
        for (size_t i = 0; i < ground->indices.size(); ++i) {
            int idx = ground->indices[i];
            const pcl::Normal& normal = normals->points[idx];
            
            // Calculate angle from vertical (dot product with up vector [0,0,1])
            // Vertical normal has z=1, so acos(normal.normal_z) gives the angle
            float angle = std::acos(std::abs(normal.normal_z)) * 180.0f / M_PI;
            
            if (angle <= max_ground_angle) {
                // Keep as ground
                refined_ground_indices.push_back(idx);
            } else {
                // Reclassify as non-ground
                additional_nonground_indices.push_back(idx);
            }
        }
        
        // Update ground indices
        ground->indices = refined_ground_indices;
        
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
        
        // Detect edges
        std::vector<int> edge_points = detectEdgePoints(cloud, ground->indices, min_height_diff, neighborhood_radius);
        std::cout << "Detected " << edge_points.size() << " edge points" << std::endl;
        
        // Create buffer around edges and reclassify
        std::vector<int> refined_ground_indices;
        std::vector<int> additional_nonground_indices;
        
        for (size_t i = 0; i < ground->indices.size(); ++i) {
            int idx = ground->indices[i];
            if (isNearEdge(cloud->points[idx], edge_points, cloud, edge_buffer)) {
                // Reclassify as non-ground
                additional_nonground_indices.push_back(idx);
            } else {
                // Keep as ground
                refined_ground_indices.push_back(idx);
            }
        }
        
        // Update ground indices
        ground->indices = refined_ground_indices;
        
        // Re-extract ground and non-ground clouds with refined classification
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
        // In a full implementation, you would process at multiple scales and use voting
        
        // For demonstration, we'll use a simple approach where points classified as ground
        // at all scales are kept as ground
        std::vector<std::vector<int>> scale_ground_indices;
        
        for (const float& scale : scales) {
            pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> scale_pmf;
            scale_pmf.setInputCloud(cloud);
            scale_pmf.setMaxWindowSize(static_cast<int>(config["pmf"]["max_window_size"].as<int>() * scale));
            scale_pmf.setSlope(config["pmf"]["slope"].as<float>());
            scale_pmf.setInitialDistance(config["pmf"]["initial_distance"].as<float>() * scale);
            scale_pmf.setMaxDistance(config["pmf"]["max_distance"].as<float>() * scale);
            
            pcl::PointIndicesPtr scale_ground(new pcl::PointIndices);
            scale_pmf.extract(scale_ground->indices);
            scale_ground_indices.push_back(scale_ground->indices);
        }
        
        // Count votes for each point
        std::map<int, int> ground_votes;
        for (const auto& indices : scale_ground_indices) {
            for (const int& idx : indices) {
                ground_votes[idx]++;
            }
        }
        
        // Determine final ground points based on voting
        std::vector<int> final_ground_indices;
        for (const auto& vote : ground_votes) {
            float vote_ratio = static_cast<float>(vote.second) / scales.size();
            if (vote_ratio >= voting_threshold) {
                final_ground_indices.push_back(vote.first);
            }
        }
        
        // Update ground indices
        ground->indices = final_ground_indices;
        
        // Re-extract ground and non-ground clouds with refined classification
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
        
        // This is a simplified implementation of context-aware filtering
        // In a full implementation, you would analyze local regions more comprehensively
        
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud);
        
        std::vector<int> refined_ground_indices;
        
        for (const int& idx : ground->indices) {
            const pcl::PointXYZ& point = cloud->points[idx];
            
            // Find neighbors within region
            std::vector<int> neighbor_indices;
            std::vector<float> neighbor_distances;
            kdtree.radiusSearch(point, region_size, neighbor_indices, neighbor_distances);
            
            // Count how many neighbors are classified as ground
            int ground_count = 0;
            for (const int& neighbor_idx : neighbor_indices) {
                if (std::find(ground->indices.begin(), ground->indices.end(), neighbor_idx) != ground->indices.end()) {
                    ground_count++;
                }
            }
            
            // Calculate ratio of ground points in neighborhood
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
            
            // If height-to-width ratio is high, it might be a wall
            // If ground ratio is low, it might be isolated ground points under a wall
            if (height_to_width_ratio > wall_detection_threshold && ground_ratio < 0.5f) {
                // Skip (will be classified as non-ground)
            } else {
                refined_ground_indices.push_back(idx);
            }
        }
        
        // Update ground indices
        ground->indices = refined_ground_indices;
        
        // Re-extract ground and non-ground clouds with refined classification
        extract.setIndices(ground);
        extract.setNegative(false);
        extract.filter(*ground_cloud);
        
        extract.setNegative(true);
        extract.filter(*nonground_cloud);
        
        std::cout << "After context-aware filtering: " << ground_cloud->size() << " ground points, " 
                  << nonground_cloud->size() << " non-ground points" << std::endl;
    }

    // Save the final ground and non-ground point clouds
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

    // Euclidean clustering on non-ground points
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(nonground_cloud);
    
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(config["ec"]["clustertolerance"].as<int>(0.01));
    ec.setMinClusterSize(config["ec"]["min_cluster_size"].as<int>(5));
    ec.setMaxClusterSize(config["ec"]["max_cluster_size"].as<int>(200));
    ec.setSearchMethod(tree);
    ec.setInputCloud(nonground_cloud);
    ec.extract(cluster_indices);
    
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
        std::vector<int> edge_points = detectEdgePoints(cloud, ground->indices, min_height_diff, neighborhood_radius);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud(new pcl::PointCloud<pcl::PointXYZ>());
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
    
    // Create an interactive viewer
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    //     std::this_thread::sleep_for(100ms);
    // }
    
    return 0;
}