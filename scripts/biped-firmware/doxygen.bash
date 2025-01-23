#!/usr/bin/env bash

# Go to script directory
cd "$(dirname $0)"

# Source dependencies
source "global.bash"

function main()
{
	# Generate Doxygen
	if [ ! -d "$project_dir_doxygen/html" ]
	then
		echo "[INFO]: Generating \"$project_name\" Doxygen..."
		cd "$project_dir"
		doxygen "$project_dir_doxygen/Doxyfile"
		assert_error $? "Failed to generate \"$project_name\" Doxygen"

		echo "[INFO]: Finished generating \"$project_name\" Doxygen."
	else
		echo "[INFO]: \"$project_name\" Doxygen exists."
	fi

	# Open generated Doxygen in browser
	if [[ "$OSTYPE" == "linux"* ]]
	then
		echo "[INFO]: Opening generated Doxygen in browser..."

		# Open generated Doxygen in browser for Linux
		google-chrome "file://$(readlink -f $project_dir_doxygen/html/index.html)" &> /dev/null || \
			open "file://$(readlink -f $project_dir_doxygen/html/index.html)" &> /dev/null
		assert_error $? "Failed to open generated Doxygen in browser"
	elif [[ "$OSTYPE" == "darwin"* ]]
	then
		echo "[INFO]: Opening generated Doxygen in browser..."

		# Open generated Doxygen in browser for macOS
		open -a Safari "file://$(readlink -f $project_dir_doxygen/html/index.html)"
		assert_error $? "Failed to open generated Doxygen in browser"
	fi
}

# Call main function
main "$@"
