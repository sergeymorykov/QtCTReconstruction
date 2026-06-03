; -- CTReconstruction Installer (Consumer-only build) --
; Перед сборкой установщика нужно выполнить scripts\build_staging.ps1,
; который наполнит dist\app\ только файлами Consumer'а и переименует
; QtCTReconstructionConsumer.exe -> CTReconstruction.exe.

#define MyAppName        "CTReconstruction"
#define MyAppVersion     "1.0.0"
#define MyAppPublisher   "Казанский федеральный университет"
#define MyAppExeName     "CTReconstruction.exe"

[Setup]
; ВНИМАНИЕ: AppId менять только при создании НОВОЙ программы.
; Один AppId = одна запись в "Программы и компоненты".
AppId={{A8F2C9D4-1234-5678-9ABC-DEF012345678}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=no
OutputDir=output
OutputBaseFilename={#MyAppName}-{#MyAppVersion}-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
SetupIconFile=..\resources\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
LicenseFile=..\LICENSE.txt
; Минимум Windows 10 64-bit (CUDA 13.2 не работает на старых ОС)
MinVersion=10.0.17763

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Дополнительные значки:"

[Files]
; Все файлы Consumer-сборки из staging-папки app\.
; QtCTReconstructionConsumer.exe вынесен отдельной строкой с DestName, чтобы
; пользователь увидел в Program Files именно CTReconstruction.exe.
Source: "app\QtCTReconstructionConsumer.exe"; DestDir: "{app}"; DestName: "{#MyAppExeName}"; Flags: ignoreversion

; Всё остальное содержимое app\ копируем как есть, исключая исходный .exe
; (он уже скопирован выше с новым именем) и любые случайные Producer-артефакты.
Source: "app\*"; DestDir: "{app}"; \
    Excludes: "QtCTReconstructionConsumer.exe,QtCTReconstruction.exe,*.pdb,*.exp,*.lib,*.ilk"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

; VC++ Redistributable — нужен для запуска, не остаётся в {app}.
Source: "prereq\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\{#MyAppName}";          Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Удалить {#MyAppName}";  Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}";    Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Тихая установка VC++ Redist, только если ещё не стоит.
Filename: "{tmp}\vc_redist.x64.exe"; \
    Parameters: "/install /quiet /norestart"; \
    StatusMsg: "Установка Microsoft Visual C++ Runtime..."; \
    Check: NeedsVCRedist

; Запустить программу после установки (опционально, по чекбоксу).
Filename: "{app}\{#MyAppExeName}"; Description: "Запустить {#MyAppName}"; \
    Flags: nowait postinstall skipifsilent

[Code]
// ---- Проверка наличия VC++ Redistributable 2015-2022 ----
function NeedsVCRedist: Boolean;
var
  Installed: Cardinal;
begin
  Result := not RegQueryDWordValue(HKEY_LOCAL_MACHINE,
    'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed)
    or (Installed <> 1);
end;

// ---- Проверка драйвера NVIDIA при старте установщика ----
function InitializeSetup: Boolean;
var
  CudaDriver: String;
  Answer: Integer;
begin
  Result := True;
  CudaDriver := ExpandConstant('{sys}\nvcuda.dll');
  if not FileExists(CudaDriver) then
  begin
    Answer := MsgBox(
      'На этом компьютере не обнаружен драйвер NVIDIA (nvcuda.dll).' + #13#10 + #13#10 +
      'Программа использует CUDA для GPU-ускорения. Без драйвера будут работать' + #13#10 +
      'только CPU-бэкенды (Serial, OpenMP) — реконструкция будет медленнее.' + #13#10 + #13#10 +
      'Рекомендуется установить актуальный драйвер NVIDIA с сайта:' + #13#10 +
      'https://www.nvidia.ru/Download/index.aspx?lang=ru' + #13#10 + #13#10 +
      'Продолжить установку без CUDA-ускорения?',
      mbConfirmation, MB_YESNO);
    if Answer = IDNO then
      Result := False;
  end;
end;
