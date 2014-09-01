Data should be stored in a list of dictionaries and saved using the module `pickle`. Each entry
in the list corresponds to a cell and should contain the keys `calcium`, `spikes`, and `fps`.

	# preprocess data
	./code/experiments/preprocess.py -i data.pck -o data_preprocessed.pck -m data_preprocessed.mat

The `-m` flag can optionally be used to get the preprocessed data in Matlab format.

	# train model on data
	./code/experiments/train.py -d data_preprocessed.pck -o model.xpck

Right now there's no script to take a trained model and produce predictions for an
arbitrary dataset. But you can take a look at

	./code/experiments/predict.py -d data_preprocessed.pck -m model.xpck

which should already produce predictions, it just doesn't save them.

The script

	./code/experiments/generalize.py -z data_train.pck -s data_test.pck -o predictions.xpck

will train a model on the preprocessed data `data_train.pck` and produce predictions for
the preprocessed data in `data_test.pck`.

The script

	./code/experiments/leave_one_out.py

takes the same parameters as `train.py`. It will train the model on all but the n-th cell
and then produce predictions for the n-th cell. It will repeat this for all n and then
save the predictions.
