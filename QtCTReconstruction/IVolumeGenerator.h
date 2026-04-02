#pragma once

// ВАЖНО: Generator3D.h должен быть включён ПЕРЕД этим файлом в наследниках,
// чтобы тип Generator3D::Params был полным. В самом этом файле мы делаем
// только forward declaration — это разрывает циклическую зависимость.
// Generator3D.h: forward-declare class IVolumeGenerator (не включает этот файл)
// Generator3DCPU.h/Generator3DGPU.h: #include "Generator3D.h" + #include "IVolumeGenerator.h"

#include "CTTypes.h"
#include <string>

namespace ct {

class Generator3D; // forward declaration — полный тип из Generator3D.h

class IVolumeGenerator {
public:
    virtual ~IVolumeGenerator() = default;

    virtual std::string name() const = 0;
    virtual bool isAvailable() const = 0;

    // Generator3D::Params используется здесь как forward-declared тип.
    // Это работает т.к. вложенный тип Params берётся через forward-declare Generator3D.
    // Компилятор увидит полный тип через Generator3D.h, включённый в .h наследников.
    virtual Volume generateBrainHU(const Generator3D::Params& params) = 0;
};

} // namespace ct
