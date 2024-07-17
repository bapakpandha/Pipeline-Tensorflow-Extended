# Submission Final: Detektor Clickbait Berbasis Machine Learning
Nama: Ahmad Rifqi Maulana

Username dicoding: si_tomb

<div style="display:flex;width:100vw;>
<"div style="position:absolute; width:75%">
  <img src="https://raw.githubusercontent.com/bapakpandha/Pipeline-Tensorflow-Extended/main/sitomb-grafana-dashboard.png" alt="Clcikbait_monitoring" style="margin-left: auto;margin-right: auto ; display:block;width=250px;" />
</div>
</div>

# Clickbait Prediction TensorFlow Extended (TFX) Pipeline

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Pipeline Stages](#pipeline-stages)
    - [Data Ingestion](#data-ingestion)
    - [Data Validation](#data-validation)
    - [Data Preprocessing](#data-preprocessing)
    - [Model Training](#model-training)
4. [Model Architecture](#model-architecture)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Model Performance](#model-performance)
7. [Deployment](#deployment)
8. [Monitoring](#monitoring)
9. [Additional Features](#additional-features)
10. [Screenshots](#screenshots)
11. [Acknowledgements](#acknowledgements)

## Introduction
This project is the final submission for the Clickbait Detector using machine learning. It demonstrates the entire pipeline from data ingestion to model deployment using TensorFlow Extended (TFX).

## Dataset
We use the [News Clickbait Dataset](https://www.kaggle.com/datasets/vikassingh1996/news-clickbait-dataset) from Kaggle for training and testing the model.

## Pipeline Stages

### Data Ingestion
**ExampleGen:** The initial stage of data processing, responsible for ingesting data into the pipeline.

### Data Validation
**StatisticGen:** Generates summary statistics for each feature in the dataset for review.  
**SchemaGen:** Uses statistics from StatisticGen to infer a schema defining the data structure, ensuring data consistency.  
**ExampleValidator:** Checks data quality by identifying missing data or values that don't conform to the schema.

### Data Preprocessing
**Transform:** Transforms the dataset by normalizing the data, modifying data patterns, and adding necessary transformations to ensure compatibility with machine learning requirements.

### Model Training
Uses the preprocessed dataset to train the model.

## Model Architecture
The model consists of 8 layers:
1. InputLayer
2. tf.reshape layer
3. text_vectorization layer
4. Embedding layer
5. Pooling 1D layer
6. Dense layer 1 (obtained through hyperparameter tuning)
7. Dense layer 2 (obtained through hyperparameter tuning)
8. Dense layer 3 (obtained through hyperparameter tuning)

Hyperparameter tuning is employed to find the optimal values for the Dense layers.

## Hyperparameter Tuning
Performed using TensorFlow's Tuner, focusing on maximizing `val_binary_accuracy` over 10 trials.

## Model Performance
The model achieved a binary accuracy of 99.5% based on testing.

## Deployment
The model is deployed using TensorFlow Serving on a Virtual Machine provided by Lintasarta Cloudeka (Deka-Flexi).  
[Clickbait Prediction Metadata](http://103.190.215.122:8501/v1/models/clickbait_prediction/metadata)  
[Public Dashboard Grafana](http://103.190.215.122:3000/public-dashboards/bdb53051b7454a4a8c1a74595875fe1b)

## Monitoring
Prometheus is used to monitor system metrics, especially the number of prediction requests in real-time. Grafana visualizes these metrics.

## Additional Features
- **Automatic Hyperparameter Tuning:** Utilizes Tuner for automatic hyperparameter optimization.
- **Clean Code Principles:** Applies clean code practices in constructing the machine learning pipeline.
- **Jupyter Notebook:** Includes a notebook for testing and sending prediction requests to the deployed system.
- **Synchronizing Prometheus and Grafana:** Enhances monitoring dashboard visualization.

## Screenshots
### Grafana Dashboard
![Grafana Dashboard](https://raw.githubusercontent.com/bapakpandha/Pipeline-Tensorflow-Extended/main/sitomb-grafana-dashboard.png)
_Description: Screenshot of the Grafana dashboard showing various metrics._

### Prometheus Monitoring
![Prometheus Monitoring](https://raw.githubusercontent.com/bapakpandha/Pipeline-Tensorflow-Extended/main/sitomb-monitoring.png)
_Description: Screenshot of Prometheus monitoring system in action._

### TensorFlow Serving Deployment
![TensorFlow Serving Deployment](https://raw.githubusercontent.com/bapakpandha/Pipeline-Tensorflow-Extended/main/sitomb-tensorflow-serving.png)
_Description: Screenshot of TensorFlow Serving deployment._

### Web App Interface
![Web App Interface](https://raw.githubusercontent.com/bapakpandha/Pipeline-Tensorflow-Extended/main/sitomb-testing.ipynb)
_Description: Screenshot of the web app interface for Clickbait Prediction._

## Acknowledgements
- Special thanks to [Kaggle](https://www.kaggle.com) for providing the dataset.
- Thanks to Lintasarta Cloudeka for cloud services.



| | Deskripsi |
| ----------- | ----------- |
| Dataset | [News Clickbait Dataset](https://www.kaggle.com/datasets/vikassingh1996/news-clickbait-dataset) |
| Masalah | Dalam era digital yang terus berkembang, praktik clickbait menjadi masalah serius dengan menyesatkan pembaca melalui headline yang menarik namun menyesatkan, seringkali mengurangi kepercayaan terhadap media dan menyebarkan misinformasi. Proyek ini bertujuan untuk mengembangkan sebuah model machine learning yang mampu mendeteksi headline clickbait. |
| Solusi machine learning | Sistem klasifikasi otomatis berbasis Machine Learning ini dapat membedakan antara headline yang merupakan clickbait dan yang bukan clickbait. Sistem ini akan menggunakan teknik pemrosesan bahasa alami (NLP) untuk menganalisis pola teks dalam judul artikel dan mempelajari fitur-fitur yang menandakan apakah sebuah judul cenderung menjadi clickbait atau bukan. Sistem ini diharapkan dapat mengurangi permasalahan penyesatan isi berita akibat headline yang bersifat clickbait |
| Metode pengolahan | Setidaknya, terdapat 6 Tahapan dalam pengolahan data pada sistem ini: <br><br>1. **Data Ingestion: ExampleGen**, merupakan tahapan awal dari pengolahan data yang berperan dalam pengambilan data ke dalam sistem pipeline <br>2. **Data Validation: StatisticGen**, pembuatan summary statistic dari tiap fitur dalam dataset yang digunakan untuk di tinjau <br>3. **Data Validation: SchemaGen**, menerima masukan berupa statistik yang dihasilkan oleh StatisticsGen, kemudian menginfers sebuah skema yang mendefinisikan struktur data, yang digunakan untuk memastikan konsistensi data.<br>4. **Data Validation: Example Validator**, melakukan pengecekan kualitas data, mengidentifikasi data yang hilang atau nilai yang tidak sesuai dengan skema.<br>5. **Data Preprocessing: Transform**, melakukan transformasi dataset dengan tujuan menormalisasikan data, mengubah pola data dengan menambahkan _xf pada kelasnya agar sesuai dengan prequistion machine learning, dan lain-lain.<br>6. **Training Data**, Proses pelatihan sistem dengan menggunakan dataset yang sebelumnya sudah diolah.|
| Arsitektur model | Model menggunakan arsitektur yang terdiri dari 8 buah layer, yaitu: InputLayer, tf.reshape layer, text_vectorization layer, Embedding layer, Pooling 1D layer, dan 3 layer Dense. Adapun 3 layer Dense didapatkan dengan cara tuning otomatis melalui hyperparameter tuning untuk mendapatkan value units terbaiknya.|
| Metrik evaluasi | Proses penentuan layer terbaik dilakukan dengan menggunakan bantuan hyperparameter tuning dengan menitikberatkan variabel val_binary_accuracy paling maksimum yang didapatkan selama 10 kali trial. Sedangkan saat training, variabel yang dititikberatkan adalah binary_accuracy maksimum |
| Performa model | Berdasarkan pengujian, hasil performa model ini mencapai akurasi 99.5% (binary_accuracy) |
| Opsi deployment | Deployment pada sistem ini menggunakan layanan Virtual Machine berbasis cloud dari Lintasarta Cloudeka (Deka-Flexi)|
| Web app | [[Clickbait Prediction Metadata]](http://103.190.215.122:8501/v1/models/clickbait_prediction/metadata) [[Public Dashboard Grafana]](http://103.190.215.122:3000/public-dashboards/bdb53051b7454a4a8c1a74595875fe1b)|
| Monitoring | Deploy model clickbait prediction menggunakan TensorFlow Serving di Virtual Machine berjalan di Layanan Cloud Computing Lintasarta Cloudeka (DEKA FLEXI), serta menggunakan Prometheus untuk memonitor sistem, terutama dalam hal jumlah permintaan prediksi secara real-time. Grafana digunakan untuk visualisasi monitoring, yang mencakup metrik jumlah permintaan prediksi TensorFlow Serving. |
