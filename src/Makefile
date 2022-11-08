DIR := '../out/io-samples'
ZOO_DIR := '../out/zoo'

evo:
	python3 -u evo.py

lindenmayer:
	python3 -u lindenmayer.py $(DIR)
	$(MAKE) convert_ps

datagen:
	$(MAKE) clean
	mkdir -p $(DIR)
	python3 -u datagen.py
	$(MAKE) convert_svg
	mkdir -p $(DIR)/svgs
	mkdir -p $(DIR)/pngs
	mv $(DIR)/*.svg $(DIR)/svgs
	mv $(DIR)/*.png $(DIR)/pngs

datagen_remote:
	$(MAKE) clean
	./scripts/conda-activate.sh
	mkdir -p $(DIR)
	xvfb-run python3 -u datagen.py 'handcoded' 100_000 1 5 2 5 5 15 5 6 $(DIR)

convert_svg:
	./scripts/svg-to-png.sh $(DIR)

convert_ps:
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
