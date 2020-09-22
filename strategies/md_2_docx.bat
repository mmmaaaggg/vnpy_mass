@echo off
:: 遍历当前文件夹下的所有后缀名为md的文件
for /f %%a in ('dir /b *.md') do (
    :: 执行pandoc命令，把每个md文件都转为docx文件，docx文件的文件名为：md文件名.md.docx
    pandoc %%a -o %%a.docx
)
pause