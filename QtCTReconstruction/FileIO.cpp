#include "FileIO.h"

#include "Utils.h"

#include <Windows.h>

#include <cstdint>
#include <fstream>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>

#include <QImage>
#include <QString>

namespace ct {

namespace {

bool parseShapeXYZ(const std::string& header, size_t& sx, size_t& sy, size_t& sz) {
    const size_t l = header.find('(');
    const size_t r = header.find(')', l == std::string::npos ? 0 : l + 1);
    if (l == std::string::npos || r == std::string::npos || r <= l + 1) {
        return false;
    }
    std::string shape = header.substr(l + 1, r - l - 1);
    for (char& c : shape) {
        if (c == ',') {
            c = ' ';
        }
    }
    std::istringstream iss(shape);
    return static_cast<bool>(iss >> sx >> sy >> sz);
}

bool parseShape2D(const std::string& header, size_t& r, size_t& c) {
    size_t l = header.find('(');
    size_t right = header.find(')', l == std::string::npos ? 0 : l + 1);
    if (l == std::string::npos || right == std::string::npos || right <= l + 1) {
        return false;
    }
    std::string shape = header.substr(l + 1, right - l - 1);
    for (char& ch : shape) {
        if (ch == ',') ch = ' ';
    }
    std::istringstream iss(shape);
    return static_cast<bool>(iss >> r >> c);
}

} // namespace

bool FileIO::saveVolumeNPY(const Volume& volume, const std::string& path_utf8) {
    const size_t z = volume.depth;
    const size_t y = volume.height;
    const size_t x = volume.width;
    if (x == 0 || y == 0 || z == 0) return false;

    std::ofstream out(path_utf8, std::ios::binary);
    if (!out.is_open()) return false;

    const std::string magic = "\x93NUMPY";
    out.write(magic.data(), magic.size());

    const unsigned char version[2] = {1, 0};
    out.write(reinterpret_cast<const char*>(version), 2);

    std::ostringstream header_ss;
    header_ss << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << z << ", " << y << ", " << x << "), }";
    std::string header = header_ss.str();
    size_t padded_len = header.size() + 1;
    while ((10 + padded_len) % 16 != 0) {
        ++padded_len;
    }
    header.append(padded_len - header.size() - 1, ' ');
    header.push_back('\n');

    const uint16_t header_len = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header.data(), header.size());

    out.write(reinterpret_cast<const char*>(volume.data.data()), x * y * z * sizeof(float));
    return out.good();
}

bool FileIO::loadVolumeNPY(const std::string& path_utf8, Volume& out_volume, const bool input_is_xyz_layout) {
    std::ifstream in(path_utf8, std::ios::binary);
    if (!in.is_open()) return false;

    char magic[6]{};
    in.read(magic, 6);
    if (!in.good() || std::string(magic, 6) != "\x93NUMPY") return false;

    unsigned char version[2]{};
    in.read(reinterpret_cast<char*>(version), 2);
    if (!in.good()) return false;

    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t hl16 = 0;
        in.read(reinterpret_cast<char*>(&hl16), sizeof(hl16));
        header_len = hl16;
    } else {
        in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    }
    if (!in.good() || header_len == 0) return false;

    std::string header(header_len, '\0');
    in.read(header.data(), header.size());
    if (!in.good()) return false;

    if (header.find("<f4") == std::string::npos) return false;

    size_t sx = 0, sy = 0, sz = 0;
    if (!parseShapeXYZ(header, sx, sy, sz) || sx == 0 || sy == 0 || sz == 0) return false;

    std::vector<float> raw(sx * sy * sz, 0.0f);
    in.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(float));
    if (!in.good()) return false;

    if (input_is_xyz_layout) {
        out_volume.assign(sx, sy, sz, 0.0f);
        #pragma omp parallel for schedule(static)
        for (int xi = 0; xi < static_cast<int>(sx); ++xi) {
            for (size_t y = 0; y < sy; ++y) {
                for (size_t z = 0; z < sz; ++z) {
                    out_volume.at(xi, y, z) = raw[(xi * sy + y) * sz + z];
                }
            }
        }
    } else {
        // z, y, x
        out_volume.assign(sz, sy, sx, 0.0f); // x = sz, y = sy, z = sx
        #pragma omp parallel for schedule(static)
        for (int zi = 0; zi < static_cast<int>(sx); ++zi) {
            for (size_t y = 0; y < sy; ++y) {
                for (size_t x = 0; x < sz; ++x) {
                    out_volume.at(x, y, zi) = raw[(zi * sy + y) * sz + x];
                }
            }
        }
    }

    out_volume.x_coords = utils::linspace(-100.0f, 100.0f, out_volume.width);
    out_volume.y_coords = utils::linspace(-100.0f, 100.0f, out_volume.height);
    out_volume.z_coords = utils::linspace(-100.0f, 100.0f, out_volume.depth);
    return true;
}

bool FileIO::savePointCloudNPY(const PointCloud& cloud, const std::string& path_utf8) {
    if (cloud.empty()) return false;
    std::ofstream out(path_utf8, std::ios::binary);
    if (!out.is_open()) return false;

    const std::string magic = "\x93NUMPY";
    out.write(magic.data(), magic.size());
    const unsigned char version[2] = {1, 0};
    out.write(reinterpret_cast<const char*>(version), 2);

    std::ostringstream header_ss;
    header_ss << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << cloud.size() << ", 4), }";
    std::string header = header_ss.str();
    size_t padded_len = header.size() + 1;
    while ((10 + padded_len) % 16 != 0) ++padded_len;
    header.append(padded_len - header.size() - 1, ' ');
    header.push_back('\n');

    uint16_t header_len = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header.data(), header.size());

    out.write(reinterpret_cast<const char*>(cloud.data()), cloud.size() * sizeof(Point));
    return out.good();
}

bool FileIO::loadPointCloudNPY(const std::string& path_utf8, PointCloud& out_cloud) {
    std::ifstream in(path_utf8, std::ios::binary);
    if (!in.is_open()) return false;

    char magic[6]{};
    in.read(magic, 6);
    if (!in.good() || std::string(magic, 6) != "\x93NUMPY") return false;

    unsigned char version[2]{};
    in.read(reinterpret_cast<char*>(version), 2);
    if (!in.good()) return false;

    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t hl16 = 0;
        in.read(reinterpret_cast<char*>(&hl16), sizeof(hl16));
        header_len = hl16;
    } else {
        in.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    }
    if (!in.good() || header_len == 0) return false;

    std::string header(header_len, '\0');
    in.read(header.data(), header.size());
    if (!in.good()) return false;

    if (header.find("<f4") == std::string::npos) return false;

    size_t rows = 0, cols = 0;
    if (!parseShape2D(header, rows, cols) || rows == 0 || cols != 4) return false;

    out_cloud.resize(rows);
    in.read(reinterpret_cast<char*>(out_cloud.data()), rows * sizeof(Point));
    
    return in.good();
}

bool FileIO::saveSlicePNG(const Slice& slice, const std::wstring& path_wide, const float min_v, const float max_v) {
    if (slice.empty() || slice.width == 0 || slice.height == 0) return false;

    const int width = static_cast<int>(slice.width);
    const int height = static_cast<int>(slice.height);
    
    QImage img(width, height, QImage::Format_Grayscale8);
    
    const float range = (max_v - min_v > 1e-6f) ? (max_v - min_v) : 1.0f;
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        uchar* rowPtr = img.scanLine(y);
        for (int x = 0; x < width; ++x) {
            const float v = utils::clamp((slice[static_cast<size_t>(height - 1 - y)][static_cast<size_t>(x)] - min_v) / range, 0.0f, 1.0f);
            rowPtr[x] = static_cast<uchar>(v * 255.0f);
        }
    }

    return img.save(QString::fromStdWString(path_wide), "PNG");
}

bool FileIO::saveSliceBMP(const Slice& slice, const std::wstring& path_wide, const float min_v, const float max_v) {
    if (slice.empty() || slice.width == 0 || slice.height == 0) {
        return false;
    }

    const int width = static_cast<int>(slice.width);
    const int height = static_cast<int>(slice.height);
    const int stride = ((width * 3 + 3) / 4) * 4;
    std::vector<unsigned char> buffer(static_cast<size_t>(stride * height), 0);

    const float range = (max_v - min_v > 1e-6f) ? (max_v - min_v) : 1.0f;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float v = utils::clamp((slice[static_cast<size_t>(height - 1 - y)][static_cast<size_t>(x)] - min_v) / range, 0.0f, 1.0f);
            const unsigned char c = static_cast<unsigned char>(v * 255.0f);
            const int idx = y * stride + x * 3;
            buffer[static_cast<size_t>(idx + 0)] = c;
            buffer[static_cast<size_t>(idx + 1)] = c;
            buffer[static_cast<size_t>(idx + 2)] = c;
        }
    }

    BITMAPFILEHEADER bfh{};
    BITMAPINFOHEADER bih{};
    bih.biSize = sizeof(BITMAPINFOHEADER);
    bih.biWidth = width;
    bih.biHeight = height;
    bih.biPlanes = 1;
    bih.biBitCount = 24;
    bih.biCompression = BI_RGB;
    bih.biSizeImage = static_cast<DWORD>(buffer.size());

    bfh.bfType = 0x4D42;
    bfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    bfh.bfSize = bfh.bfOffBits + bih.biSizeImage;

    std::ofstream out(path_wide, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    out.write(reinterpret_cast<const char*>(&bfh), sizeof(bfh));
    out.write(reinterpret_cast<const char*>(&bih), sizeof(bih));
    out.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    return out.good();
}

} // namespace ct
