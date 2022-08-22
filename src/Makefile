lindenmayer:
	python3 lindenmayer.py
	$(MAKE) convert

datagen:
	python3 datagen.py 1 1 1 10 10 3 5 True '../out/'
	$(MAKE) convert

convert:
	./scripts/convert-ps.sh	# Convert ps -> png
	rm ../imgs/*.ps		# Remove leftover ps

clean:
	rm ../imgs/*.png
	rm ../imgs/*grammar.txt

view:
	sxiv ../imgs

grammars:
	./scripts/show_;grammars.sh
