#pragma once

#include "CTTypes.h"

#include <string>

namespace ct {

class FileIO {
public:
    static bool saveVolumeNPY(const Volume& volume, const std::string& path_utf8);
    static bool loadVolumeNPY(const std::string& path_utf8, Volume& out_volume, bool input_is_xyz_layout);
    
    // Функции для работы с облаком точек
    static bool savePointCloudNPY(const PointCloud& cloud, const std::string& path_utf8);
    static bool loadPointCloudNPY(const std::string& path_utf8, PointCloud& out_cloud);

    // Экспорт в PNG (8-bit grayscale)
    static bool saveSlicePNG(const Slice& slice, const std::wstring& path_wide, float min_v, float max_v);

    static bool saveSliceBMP(const Slice& slice, const std::wstring& path_wide, float min_v, float max_v);
};

} // namespace ct
