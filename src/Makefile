ROOT := '../out/'
DIR := '../out/sep27/fitted'
ZOO_DIR := '../out/zoo'

lindenmayer:
	python3 -u lindenmayer.py $(DIR)
	$(MAKE) convert

datagen:
	$(MAKE) clean
	mkdir -p $(DIR)
# n_grammars, axiom_length[2], n_rules[2], rule_length[2], n_specimens, depth, out_dir
	python3 -u datagen.py 'handcoded' 5 1 5 2 3 4 10 3 3 $(DIR)
	$(MAKE) convert

datagen_remote:
	$(MAKE) clean
	./scripts/conda-activate.sh
	mkdir -p $(DIR)
	xvfb-run python3 -u datagen.py 'handcoded' 100_000 1 5 2 5 5 15 5 6 $(DIR)

convert:
	@echo "Converting .ps files..."
	./scripts/convert-ps.sh $(DIR)
	@echo "Removing leftover .ps files..."
	@rm -f $(DIR)/*.ps	

clean:
	@rm -f $(DIR)/*.svg
	@rm -f $(DIR)/*.png
	@rm -f $(DIR)/*.ps
	@rm -f $(DIR)/*.dat
	@rm -f $(DIR)/*.log

view:
	geeqie $(DIR)

grammars:
	less $(DIR)/grammars.log

zoo:
	@rm -f $(ZOO_DIR)/*.svg
	python3 -u zoo.py $(ZOO_DIR)
