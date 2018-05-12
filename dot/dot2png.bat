echo off
rem This is convert dot format file to png graph
rem need graphviz <https://www.graphviz.org/>
rem Change dot.exe directory as your environment
rem 
rem usage an example to convert tree1.dot to tree1.png
rem                dot2png.bat tree1
rem
echo on
C:\python3\graphviz-2.38\release\bin\dot.exe -Kdot -Tpng %1.dot -o%1.png
