// Consumer build entry point.
// Идентичен main.cpp (Producer), но грузит main_consumer.qml через свой QML-модуль
// URI ("QtCTReconstructionConsumer"). Compile-time дефайн CT_CONSUMER_BUILD
// (выставлен из CMake) включает в контроллере Q_INVOKABLE loadSinograms и
// меняет ветку reconstructionTask.

#include <QApplication>
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
#ifdef _WIN32
    SetConsoleOutputCP(1251);
#endif

    qputenv("QT_QUICK_CONTROLS_STYLE", "Fusion");
    qputenv("QSG_RHI_BACKEND", "opengl");
    qputenv("QT3D_RENDERER", "opengl");

    QApplication app(argc, argv);

    // Та же иконка приложения, что у Producer (см. main.cpp).
    {
        const QString exeDir = QCoreApplication::applicationDirPath();
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
                    qDebug() << "[CT] icon loaded from:" << candidate
                             << "sizes:" << ic.availableSizes();
                    loaded = true;
                    break;
                }
            }
        }
        if (!loaded) {
            qDebug() << "[CT] icon NOT loaded; window title-bar icon will be Qt default."
                     << "Run 'python resources/generate_icon.py' + cmake reconfigure + rebuild.";
        }
    }

    CtReconstructionController controller;

    // Регистрация PointCloudGeometry оставлена для совместимости — Consumer её
    // не использует, но если QML случайно импортирует тип, он будет доступен.
    qmlRegisterType<PointCloudGeometry>("QtCTReconstruction", 1, 0, "PointCloudGeometry");

    QQmlApplicationEngine engine;

    CtSliceImageProvider* imageProvider = new CtSliceImageProvider(&controller);
    engine.addImageProvider(QStringLiteral("ct"), imageProvider);

    engine.rootContext()->setContextProperty(QStringLiteral("controller"), &controller);

    // QML-модуль Consumer'а зарегистрирован с URI "QtCTReconstructionConsumer"
    // (см. CMakeLists.txt → qt_add_qml_module(QtCTReconstructionConsumer ...)).
    engine.load(QUrl(QStringLiteral("qrc:/QtCTReconstructionConsumer/QtCTReconstruction/main_consumer.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
