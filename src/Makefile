lindenmayer:
	python3 lindenmayer.py
	./convert-ps.sh		# Convert ps -> png
	rm ../imgs/*.ps		# Remove leftover ps

datagen:
	python3 datagen.py
	./convert-ps.sh
	rm ../imgs/*.ps

clean:
	rm -rf ../imgs/*

show:
	sxiv ../imgs
