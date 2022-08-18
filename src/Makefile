lindenmayer:
	python3 lindenmayer.py
	./scripts/convert-ps.sh	# Convert ps -> png
	rm ../imgs/*.ps		# Remove leftover ps

datagen:
	python3 datagen.py
	./scripts/convert-ps.sh
	rm ../imgs/*.ps

clean:
	rm ../imgs/*.png
	rm ../imgs/*grammar.txt

view:
	sxiv ../imgs

grammars:
	./scripts/show_grammars.sh
