// // shm_creator_sim.cpp
// #include "ImageStreamIO.h"
// #include <iostream>
// #include <fstream>
// #include <string>
// #include <nlohmann/json.hpp>
// #include <unistd.h>
// #include <vector>
// #include <algorithm>

// using json = nlohmann::json;

// /// Compile with:
// /// g++ shm_creator_sim.cpp -o shm_creator_sim -I/home/benjamin/Documents/dcs/libImageStreamIO -L/home/rtc/Documents/dcs/libImageStreamIO -lImageStreamIO -I/usr/include/nlohmann -pthread

// int create_and_init_image(const std::string& name, int width, int height, int datatype, int nbsem) {
//     IMAGE* img = (IMAGE*) malloc(sizeof(IMAGE));
//     if (!img) {
//         std::cerr << "ERROR: malloc failed for IMAGE struct\n";
//         return -1;
//     }

//     uint32_t dims[2] = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
//     errno_t ret = ImageStreamIO_createIm_gpu(img, name.c_str(), 2, dims, datatype, -1, 1, nbsem, 0, 0);

//     if (ret != 0) {
//         std::cerr << "ERROR: Failed to create SHM " << name << "\n";
//         free(img);
//         return ret;
//     }

//     // Initialize array to zero based on datatype
//     size_t npix = static_cast<size_t>(width) * height;
//     switch (datatype) {
//         case _DATATYPE_UINT8:  std::fill_n(img->array.UI8,  npix, 0); break;
//         case _DATATYPE_UINT16: std::fill_n(img->array.UI16, npix, 0); break;
//         case _DATATYPE_UINT32: std::fill_n(img->array.UI32, npix, 0); break;
//         case _DATATYPE_INT16:  std::fill_n(img->array.SI16, npix, 0); break;
//         case _DATATYPE_INT32:  std::fill_n(img->array.SI32, npix, 0); break;
//         case _DATATYPE_FLOAT:  std::fill_n(img->array.F,    npix, 0.0f); break;
//         case _DATATYPE_DOUBLE: std::fill_n(img->array.D,    npix, 0.0); break;
//         default:
//             std::cerr << "WARNING: Unknown datatype, cannot zero-initialize.\n";
//     }

//     // Optional: post semaphore 0 if needed
//     ImageStreamIO_sempost(img, 0);

//     // Cleanup
//     ImageStreamIO_destroyIm(img);
//     free(img);

//     return 0;
// }

// int main() {
//     const std::string json_path = "/home/rtc/Documents/dcs/asgard-cred1-server/cred1_split.json";
//     std::ifstream file(json_path);
//     if (!file.is_open()) {
//         std::cerr << "ERROR: Failed to open " << json_path << "\n";
//         return 1;
//     }

//     json j;
//     file >> j;

//     const int NBsem = 10;
//     const int datatype = _DATATYPE_UINT16;

//     // Create baldr1 to baldr4
//     for (int i = 1; i <= 4; ++i) {
//         std::string key = "baldr" + std::to_string(i);
//         if (!j.contains(key)) {
//             std::cerr << "ERROR: Missing " << key << " in JSON.\n";
//             continue;
//         }

//         int xsz = j[key]["xsz"];
//         int ysz = j[key]["ysz"];
//         std::string shm_name = key;  // e.g., "baldr1" → creates "baldr1.im.shm"

//         if (create_and_init_image(shm_name, xsz, ysz, datatype, NBsem) == 0) {
//             std::cout << "Created " << shm_name << ".im.shm OK.\n";
//         }
//     }

//     // Create global frame SHM "cred1"
//     if (create_and_init_image("cred1", 320, 256, datatype, NBsem) == 0) {
//         std::cout << "Created cred1.im.shm OK.\n";
//     }

//     return 0;
// }

// shm_creator_sim.cpp
#include "ImageStreamIO.h"
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include <unistd.h>

using json = nlohmann::json;

/// compile
// g++ shm_creator_sim.cpp -o shm_creator_sim -I/home/benjamin/Documents/dcs/libImageStreamIO -L/home/benjamin/Documents/dcs/libImageStreamIO -lImageStreamIO -I/usr/include/nlohmann -pthread



int create_image(const std::string& name, int width, int height, int datatype, int nbsem) {
    
    IMAGE* img = (IMAGE*) malloc(sizeof(IMAGE));
    
    // Check if name includes "cred" (case-sensitive)
    bool is_cred = name.find("cred") != std::string::npos;

    errno_t ret;
    if (is_cred) {
        // 3D image: width x height x nrs
        const int nrs = 5; //200;  // or get this from a config/env
        uint32_t dims[3] = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            static_cast<uint32_t>(nrs)
        };

        std::cout << "Creating 3D image: " << name << " with dims (" << dims[0] << ", " << dims[1] << ", " << dims[2] << ")" << std::endl;

        ret = ImageStreamIO_createIm_gpu(img, name.c_str(), 3, dims, datatype, -1, 1, nbsem, 0, 0);
    } else {
        // 2D image
        uint32_t dims[2] = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        std::cout << "Creating 2D image: " << name << " with dims (" << dims[0] << ", " << dims[1] << ")" << std::endl;

        ret = ImageStreamIO_createIm_gpu(img, name.c_str(), 2, dims, datatype, -1, 1, nbsem, 0, 0);
    }

    if (ret != 0) {
        std::cerr << "Error creating image " << name << ", error code: " << ret << std::endl;
        free(img);
        return EXIT_FAILURE;
    }

    //IMAGE* img = (IMAGE*) malloc(sizeof(IMAGE));
    //uint32_t dims[2] = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

    //errno_t ret = ImageStreamIO_createIm_gpu(img, name.c_str(), 2, dims, datatype, -1, 1, nbsem, 0, 0);
    //ImageStreamIO_createIm_gpu(img, name.c_str(), 2, dims, datatype, -1, 1, nbsem, 0, 0);

    if (ret == 0) {
        // Zero initialize the array depending on datatype
        size_t npix = (size_t) width * height;
        switch (datatype) {
            case _DATATYPE_UINT8:
                std::fill_n(img->array.UI8, npix, 0);
                break;
            case _DATATYPE_UINT16:
                std::fill_n(img->array.UI16, npix, 0);
                break;
            case _DATATYPE_UINT32:
                std::fill_n(img->array.UI32, npix, 0);
                break;
            case _DATATYPE_INT16:
                std::fill_n(img->array.SI16, npix, 0);
                break;
            case _DATATYPE_INT32:
                std::fill_n(img->array.SI32, npix, 0);
                break;
            case _DATATYPE_FLOAT:
                std::fill_n(img->array.F, npix, 0.0f);
                break;
            case _DATATYPE_DOUBLE:
                std::fill_n(img->array.D, npix, 0.0);
                break;
            default:
                std::cerr << "WARNING: Unknown datatype, cannot zero array.\n";
        }
    }

    return ret;
}

#include <unistd.h>
#include <limits.h>
#include <string>
#include <stdexcept>

std::string get_executable_dir() {
    char buf[PATH_MAX];
    ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf)-1);
    if (len == -1) throw std::runtime_error("cannot read /proc/self/exe");
    buf[len] = '\0';
    std::string exe_path(buf);
    auto pos = exe_path.find_last_of('/');
    return exe_path.substr(0, pos);
}

std::string get_json_path() {
    std::string exe_dir = get_executable_dir();
    // binary lives in dcs/simulation → go up one, then into asgard-cred1-server
    return "/home/jcranney/git/dcs/asgard-cred1-server/cred1_split.json";
}

int main() {
    const std::string json_path = get_json_path();//"/home/rtc/Documents/dcs/asgard-cred1-server/cred1_split.json";
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open " << json_path << "\n";
        return 1;
    }

    json j;
    file >> j;

    int NBsem = 10;
    // cross check these data types with real CRED 1 server : asgard-cred1-server/asgard_ZMQ_CRED1_server.c
    int datatype_subframe = _DATATYPE_INT32;//_DATATYPE_UINT16;
    int datatype_global = _DATATYPE_UINT16; 

    for (int i = 1; i <= 4; ++i) {
        std::string key = "baldr" + std::to_string(i);
        if (!j.contains(key)) {
            std::cerr << "ERROR: Missing " << key << " in JSON.\n";
            continue;
        }
        int xsz = j[key]["xsz"];
        int ysz = j[key]["ysz"];
        //std::string shm_name = key + ".im.shm";  // FIXED: no "/dev/shm/"
        std::string shm_name = "baldr" + std::to_string(i); // dont nee the .im.shm extention - this is created automatically 
        if (create_image(shm_name, xsz, ysz, datatype_subframe, NBsem) == 0) {
            std::cout << "Created " << shm_name << " OK.\n";
        } else {
            std::cerr << "ERROR: Could not create " << shm_name << "\n";
        }
        // if (create_image(shm_name, xsz, ysz, datatype, NBsem) != 0) {
        //     std::cerr << "ERROR: Could not create " << shm_name << "\n";
        // }
    }

    // Also create global SHM
    if (create_image("cred1", 320, 256, datatype_global, NBsem) == 0) {\
        std::cout << "Created " << "cred1" << " OK.\n";
    } else{
        std::cerr << "ERROR: Could not create cred1.im.shm\n";
    }

    return 0;
}
// int main() {
//     const std::string json_path = "/home/rtc/Documents/dcs/asgard-cred1-server/cred1_split.json";
//     std::ifstream file(json_path);
//     if (!file.is_open()) {
//         std::cerr << "ERROR: Failed to open " << json_path << "\n";
//         return 1;
//     }

//     json j;
//     file >> j;

//     int NBsem = 10;
//     int datatype = _DATATYPE_UINT16;

//     // Create SHM for baldr1–4 from JSON
//     for (int i = 1; i <= 4; ++i) {
//         std::string key = "baldr" + std::to_string(i);
//         if (!j.contains(key)) {
//             std::cerr << "ERROR: Missing " << key << " in JSON.\n";
//             continue;
//         }
//         int xsz = j[key]["xsz"];
//         int ysz = j[key]["ysz"];
//         std::string shm_name = "/dev/shm/" + key + ".im.shm";
//         if (create_image(shm_name, xsz, ysz, datatype, NBsem) != 0) {
//             std::cerr << "ERROR: Could not create " << shm_name << "\n";
//         }
//     }

//     // Also create global frame SHM (fixed size 320 x 256)
//     if (create_image("/dev/shm/cred1.im.shm", 320, 256, datatype, NBsem) != 0) {
//         std::cerr << "ERROR: Could not create cred1.im.shm\n";
//     }

//     return 0;
// }