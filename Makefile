local := ~/Documents/tensorflow-adversarial/
remote := lab:/home/zzg0009/Documents/tf-adv/

get :
	rsync -abvzh \
		--delete-after \
		--exclude-from='exclude-get.txt' \
		$(remote) $(local)

put :
	rsync -abvzh \
		--exclude-from='exclude-put.txt' \
		$(local) $(remote)

clean :
	rm *~
