local := ~/Documents/tensorflow-adversarial/
remote := lab:/home/zzg0009/Documents/tf-adv/

put :
	rsync -abvzh \
		--exclude-from='exclude-put.txt' \
		$(local) $(remote)

get :
	rsync -abvzh \
		--exclude-from='exclude-get.txt' \
		$(remote) $(local)

clean :
	rm *~
