for file in `ls ./data` 
	do
        echo "tuning on $file"
        file_name=`basename $file .txt`
		python3 ./data_stream_control.py --data_file $file_name 
	done
