lindenmayer:
	python3 lindenmayer.py
	$(MAKE) convert

datagen:
# n_grammars, min_n_rules, max_n_rules, max_rule_len, n_specimens, devel_length
	python3 datagen.py 10 2 5 15 3 5
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
	./scripts/show_grammars.sh
