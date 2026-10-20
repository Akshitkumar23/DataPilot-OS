Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Always resolve the exact directory of this script file regardless of current working directory
strPath = fso.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = strPath

strExe = strPath & "\node_modules\electron\dist\electron.exe"

If fso.FileExists(strExe) Then
    WshShell.Run """" & strExe & """ """ & strPath & """", 0, False
Else
    WshShell.Run "cmd /c npx electron .", 0, False
End If
