// QApplication (не QGuiApplication!) обязателен — QFileDialog и любые другие
// QWidget-based диалоги (open/save file, message box) требуют QApplication.
// Без неё первый же QFileDialog::getOpenFileName/getExistingDirectory падает с
// фатальной ошибкой "Cannot create a QWidget without QApplication".
#include <QApplication>
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
