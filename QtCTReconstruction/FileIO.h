#pragma once

#include "CTTypes.h"

#include <string>

namespace ct {

class FileIO {
public:
    static bool saveVolumeNPY(const Volume& volume, const std::string& path_utf8);
    static bool loadVolumeNPY(const std::string& path_utf8, Volume& out_volume, bool input_is_xyz_layout);
    static bool saveSliceBMP(const Slice& slice, const std::wstring& path_wide, float min_v, float max_v);
};

} // namespace ct
