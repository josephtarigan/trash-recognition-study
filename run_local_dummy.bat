gcloud ai-platform local train --package-path trainer/ --module-name trainer.trainer.InceptionV3Trainer -- --batch-size 1 --epoch 6 --training-set-folder D:/Workspaces/python/TrashRecognition/trainer/dummy_dataset/ --output-dir D:/Workspaces/python/output/