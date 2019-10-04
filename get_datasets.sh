#!/bin/sh

wget https://meli-data-challenge.s3.amazonaws.com/train.csv.gz --directory resources
wget https://meli-data-challenge.s3.amazonaws.com/test.csv --directory resources
wget https://meli-data-challenge.s3.amazonaws.com/sample_submission.csv --directory resources
gunzip resources/train.csv.gz