lindenmayer:
	python3 lindenmayer.py
	$(MAKE) convert

datagen:
# n_grammars, min_n_rules, max_n_rules, max_rule_len, n_specimens, devel_length
	python3 datagen.py 10 2 5 15 3 5
	$(MAKE) convert

datagen_remote:
	xvfb-run python3 datagen.py 100_000 1 5 20 5 7
	$(MAKE) convert

# datagen_g2:
# 	python3 /home/djl328/prob-repl/src/datagen.py /home/djl328/prob-repl/imgs/
# 	/home/djl328/prob-repl/src/scripts/convert-ps.sh
# 	rm /home/djl328/prob-repl/imgs/*.ps

convert:
	@echo "Converting .ps files..."
	@./scripts/convert-ps.sh
	@echo "Removing leftover .ps files..."
	@rm ../imgs/*.ps	

clean:
	@rm ../imgs/*.png
	@rm ../imgs/*grammar.txt

view:
	sxiv ../imgs

grammars:
	./scripts/show_grammars.sh
