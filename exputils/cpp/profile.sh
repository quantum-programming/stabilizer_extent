set -ex

path=./exputils/cpp
filename=$path/calc_dot

echo "now compiling..."
/usr/bin/g++-10 -pg -g -fdiagnostics-color=always $filename.cpp -o $filename.exe -std=c++17 -Wall -Wextra -O2 -lz

echo "now running..."
$filename.exe
gprof $filename.exe gmon.out > gprof.profile
rm gmon.out