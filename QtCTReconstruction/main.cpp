// QApplication (не QGuiApplication!) обязателен — QFileDialog и любые другие
// QWidget-based диалоги (open/save file, message box) требуют QApplication.
// Без неё первый же QFileDialog::getOpenFileName/getExistingDirectory падает с
// фатальной ошибкой "Cannot create a QWidget without QApplication".
#include <QApplication>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QIcon>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "CtReconstructionController.h"
#include "CtSliceImageProvider.h"
#include "PointCloudGeometry.h"
#include <windows.h>

int main(int argc, char *argv[])
{
#if defined(Q_OS_WIN) && QT_VERSION_CHECK(5, 6, 0) <= QT_VERSION && QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

#ifdef _WIN32
    SetConsoleOutputCP(1251);
#endif

    // Используем Fusion-стиль — кросс-платформенный, поддерживает полную кастомизацию элементов QML
    qputenv("QT_QUICK_CONTROLS_STYLE", "Fusion");
    // Основной RHI-бэкенд для QQuickWindow
    qputenv("QSG_RHI_BACKEND", "opengl");
    // Qt3D использует собственный RHI-контекст; явно переключаем на OpenGL,
    // иначе он выбирает D3D11 по умолчанию → ошибка "Failed to create input layout"
    qputenv("QT3D_RENDERER", "opengl");

    QApplication app(argc, argv);

    // Producer build entry point. CT_PRODUCER_BUILD выставлен из CMake;
    // controller использует его для условной компиляции Producer-only методов
    // (exportAllSinograms и т.п.). Consumer-вариант — main_consumer.cpp.

    // Иконка приложения для title bar и taskbar (рантайм-уровень). Иконка .exe
    // (для Explorer / pre-launch taskbar) подключена через app_icon.rc в CMake.
    {
        // 1. Сначала перечислим, что есть в Qt-ресурсе по prefix /icons.
        qDebug() << "[CT] === icon diagnostics ===";
        const QStringList iconsInResource = QDir(":/icons").entryList();
        qDebug() << "[CT] Qt resources at :/icons :" << iconsInResource;

        // 2. Перебираем кандидатов; первый успешный — победил.
        const QString exeDir = QCoreApplication::applicationDirPath();
        qDebug() << "[CT] applicationDirPath:" << exeDir;
        bool loaded = false;
        for (const QString& candidate : {
            QStringLiteral(":/icons/icon.png"),     // Qt resource (приоритет)
            exeDir + "/icon.png",
            exeDir + "/resources/icon.png",
            exeDir + "/../../../resources/icon.png" }) {
            const bool exists = QFile::exists(candidate);
            qDebug() << "[CT] icon search:" << candidate << "exists=" << exists;
            if (exists) {
                QIcon ic(candidate);
                if (!ic.isNull()) {
                    app.setWindowIcon(ic);
                    qDebug() << "[CT] icon LOADED from:" << candidate
                             << "sizes:" << ic.availableSizes();
                    loaded = true;
                    break;
                } else {
                    qDebug() << "[CT] icon file exists but QIcon could not decode it (corrupt PNG?)";
                }
            }
        }
        if (!loaded) {
            qDebug() << "[CT] icon NOT loaded — fallback Qt default.";
            qDebug() << "[CT] Checklist:";
            qDebug() << "  1) Run 'python resources/generate_icon.py' (must produce resources/icon.png)";
            qDebug() << "  2) Re-run cmake configure (must print '[CT] App icon (.png embedded)')";
            qDebug() << "  3) Rebuild (must include this exe re-link, not just incremental)";
        }
        qDebug() << "[CT] === end icon diagnostics ===";
    }

    // Create the reconstruction controller
    CtReconstructionController controller;

    qmlRegisterType<PointCloudGeometry>("QtCTReconstruction", 1, 0, "PointCloudGeometry");

    QQmlApplicationEngine engine;

    // Add image provider for CT slice images
    CtSliceImageProvider* imageProvider = new CtSliceImageProvider(&controller);
    engine.addImageProvider(QStringLiteral("ct"), imageProvider);

    // Expose controller to QML context
    engine.rootContext()->setContextProperty(QStringLiteral("controller"), &controller);

    engine.load(QUrl(QStringLiteral("qrc:/QtCTReconstruction/QtCTReconstruction/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
