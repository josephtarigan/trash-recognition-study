gcloud ai-platform local train --package-path trainer/ --module-name trainer.InceptionV3Trainer -- --batch-size 1 --epoch 6 --training-set-folder gs://trash-recognition-project/dummy_dataset --output-dir gs://trash-recognition-project/output