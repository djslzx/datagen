# all
egg_pdf: egg_pdf_ns egg_pdf_ns_filtered egg_pdf_random
egg_cdf: egg_cdf_ns egg_cdf_ns_filtered egg_cdf_random
egg_scatter: egg_scatter_ns egg_scatter_ns_filtered egg_scatter_random

# scatter

egg_scatter_ns:
	python -u learner.py egg_scatter \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99.txt \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99-simpl.txt

egg_scatter_ns_filtered:
	python -u learner.py egg_scatter \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99.txt \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99-simpl-filtered.txt

egg_scatter_random:
	python -u learner.py egg_scatter \
	 ../datasets/random/random_100k.txt \
	 ../datasets/random/random_100k_simpl.txt

# CDF

egg_cdf_ns:
	python -u learner.py egg_cdf \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99.txt \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99-simpl.txt

egg_cdf_ns_filtered:
	python -u learner.py egg_cdf \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99.txt \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99-simpl-filtered.txt

egg_cdf_random:
	python -u learner.py egg_cdf \
	 ../datasets/random/random_100k.txt \
	 ../datasets/random/random_100k_simpl.txt

# PDF

egg_pdf_ns:
	python -u learner.py egg_pdf \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99.txt \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99-simpl.txt

egg_pdf_ns_filtered:
	python -u learner.py egg_pdf \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99.txt \
	 ../datasets/evo-bigpop/bigpop-0-gen=0-99-simpl-filtered.txt

egg_pdf_random:
	python -u learner.py egg_pdf \
	 ../datasets/random/random_100k.txt \
	 ../datasets/random/random_100k_simpl.txt
