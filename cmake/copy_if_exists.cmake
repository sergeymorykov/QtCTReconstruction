# Утилита для использования из add_custom_command(POST_BUILD).
# Копирует SRC → DST, если SRC существует. Иначе печатает мягкое предупреждение
# и завершается успехом (чтобы сборка не падала из-за отсутствия опционального
# ресурса вроде иконки).
#
# Использование:
#   cmake -DSRC=path/to/source -DDST=path/to/dest -P copy_if_exists.cmake

if(NOT DEFINED SRC OR NOT DEFINED DST)
    message(FATAL_ERROR "copy_if_exists.cmake: SRC and DST must be defined")
endif()

if(EXISTS "${SRC}")
    file(COPY_FILE "${SRC}" "${DST}" ONLY_IF_DIFFERENT)
    message(STATUS "Copied ${SRC} -> ${DST}")
else()
    message(STATUS "Skipped (source missing): ${SRC}")
endif()
