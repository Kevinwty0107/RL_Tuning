for file in `ls ./data` 
	do
        if [ "${file##*.}" = "bz2" ]; then
            echo "Unzip on $file"
            bzip2 -d ./data/$file
        fi
	done