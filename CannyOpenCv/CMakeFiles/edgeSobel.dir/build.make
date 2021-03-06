# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv"

# Include any dependencies generated for this target.
include CMakeFiles/edgeSobel.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/edgeSobel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/edgeSobel.dir/flags.make

CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o: CMakeFiles/edgeSobel.dir/flags.make
CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o: edgeSobel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o -c "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv/edgeSobel.cpp"

CMakeFiles/edgeSobel.dir/edgeSobel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/edgeSobel.dir/edgeSobel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv/edgeSobel.cpp" > CMakeFiles/edgeSobel.dir/edgeSobel.cpp.i

CMakeFiles/edgeSobel.dir/edgeSobel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/edgeSobel.dir/edgeSobel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv/edgeSobel.cpp" -o CMakeFiles/edgeSobel.dir/edgeSobel.cpp.s

CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o.requires:

.PHONY : CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o.requires

CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o.provides: CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o.requires
	$(MAKE) -f CMakeFiles/edgeSobel.dir/build.make CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o.provides.build
.PHONY : CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o.provides

CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o.provides.build: CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o


# Object files for target edgeSobel
edgeSobel_OBJECTS = \
"CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o"

# External object files for target edgeSobel
edgeSobel_EXTERNAL_OBJECTS =

edgeSobel: CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o
edgeSobel: CMakeFiles/edgeSobel.dir/build.make
edgeSobel: /usr/local/lib/libopencv_dnn.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_gapi.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_ml.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_objdetect.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_photo.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_stitching.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_video.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_calib3d.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_features2d.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_flann.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_highgui.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_videoio.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_imgcodecs.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_imgproc.so.4.1.0
edgeSobel: /usr/local/lib/libopencv_core.so.4.1.0
edgeSobel: CMakeFiles/edgeSobel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable edgeSobel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/edgeSobel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/edgeSobel.dir/build: edgeSobel

.PHONY : CMakeFiles/edgeSobel.dir/build

CMakeFiles/edgeSobel.dir/requires: CMakeFiles/edgeSobel.dir/edgeSobel.cpp.o.requires

.PHONY : CMakeFiles/edgeSobel.dir/requires

CMakeFiles/edgeSobel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/edgeSobel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/edgeSobel.dir/clean

CMakeFiles/edgeSobel.dir/depend:
	cd "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv" "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv" "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv" "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv" "/home/gaetano/Scrivania/Esercizi OpenCV/EdgeSobelOpenCv/CMakeFiles/edgeSobel.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/edgeSobel.dir/depend

