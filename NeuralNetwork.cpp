#include <iostream>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <fstream>


int main() {
	// datasetimiz ve labellarimiz.
	arma::mat dataset;
	arma::mat labels;

	// Labellarimiz ve datasetimiz ayri dosyalarda o y�zden ayri ayri y�kl�yoruz.
	bool loaded = mlpack::data::Load("Datasets/digits.csv", dataset);
	bool loaded2 = mlpack::data::Load("Datasets/labels.csv", labels);
	
	// istersek bu sekilde y�klemenin d�zg�n ger�eklesip ger�eklesmedigini kontrol edebiliriz ama zaten y�kleme aninda bir sorun olduysa mlpack bize run time error
	// atiyor.
	if (!loaded || !loaded2) {
		std::cout << "Dataset can not be loaded properly ! " << std::endl;
		return -1;
	}

	// Row ve column sayilarina bakiyoruz.
	std::cout << "Data set rows : " << dataset.n_rows << " columns : " << dataset.n_cols << std::endl;
	std::cout << "Labels rows : " << labels.n_rows << " columns : " << labels.n_cols << std::endl;


	// model objemizi olusturuyoruz.
	mlpack::ann::FFN<> model; // loss function icin default u NegativeLogLikelihood ve parametrelere baslangic degeri atamak icin RandomInitilization.
							  // istersek constructor a istedigimiz optimizer i verebiliriz ama su an i�in bu defaut degerler uygun.

	// Structure : 400 / 25 / 10
	model.Add<mlpack::ann::Linear<>>(dataset.n_rows,25); // 400 input n�ron ve output olarak 25 n�ron
	model.Add<mlpack::ann::SigmoidLayer<>>();			 // ardindan sigmoid aktivasyon fonksiyonu. normalde neural networklerde aktivasyon fonksiyonu layer olarak
														 // sayilmaz ama mlpack te �yle adlandirilmis. yani aktivasyon fonksiyonlarini ayri bir layer olarak ekliyoruz.
	model.Add<mlpack::ann::Linear<>>(25,10);			 // daha sonra diger linear layerimizi ekliyoruz. 25 inputu ve 10 outputu var.
	model.Add<mlpack::ann::LogSoftMax<>>();				 // ve son olarak 10 outputumuza logsoftmax aktivasyon fonksiyonu uyguluyoruz ��nk� birden fazla sinifi
														 // siniflandiriyoruz.
	
	model.Train(dataset,labels);	// modelimizi egitiyoruz.

	arma::mat trainingPredcitions;
	model.Predict(dataset, trainingPredcitions); // modelin accuracy ine bakmak i�in tahmin yapiyoruz.

	// tahmin yaptigimiz matrix in ka�a ka� oldugunu kontrol ediyoruz.
	std::cout << "Training Predictions : " << trainingPredcitions.n_rows << "x" << trainingPredcitions.n_cols << std::endl;

	// ilk 10 training example in her bir sinif i�in olasilik degerlerine bakiyoruz ileride bu degerlerin maksimumunu alip sinif degerimiz ne ona bakacagiz.
	for (int i = 0; i < 10; i++) {
		std::cout << trainingPredcitions(arma::span(0, trainingPredcitions.n_rows - 1), i) << "\n";
	}

	// 10 tane sinifimiz vardi. bunlarin maksimumunu almak i�in (1xm=5000) lik bos bir matrix olusturuyoruz.
	arma::mat predictions = arma::zeros<arma::mat>(1, trainingPredcitions.n_cols);

	for (size_t i = 0; i < trainingPredcitions.n_cols; i++) {
		// 10 tane olasligin i�inden maximum degere sahip olan index numarasini aliyoruz ve o index degerine 1 ekliyoruz ��nk� labels dosyasinda 0 degeri 10 olarak
		// 1 degeri 0 olarak tutuluyor yani her bir sinif 1 eksigi olarak labellanmis ve bizim 10xm lik predictions matriximizde 1'ler sinifi 0. indexte tutuluyor 
		// (0+1 = 1) ve 0'lar sinifi 9.indexte (9+1 = 10) tutuluyor.
		predictions(i) = (trainingPredcitions(arma::span(0, trainingPredcitions.n_rows - 1), i).index_max() + 1);
	}

	// istersek predictions degerlerimizi ve ger�ek labellarimzi bir dosyaya bastirip kontrol edebiliriz.
	std::ofstream myFile("predictions_labels.txt");

	myFile << "Predictions , Labels\n";
	for (size_t i = 0; i < predictions.n_cols; i++) {
		myFile << predictions(i)<<","<<labels(i) << "\n";
	}

	myFile.close();
	
	// son olarak da accuracy imizi hesapliyoruz.
	size_t correct = arma::accu(predictions == labels);
	double trainingAccuracy = (double(correct) / labels.n_cols) * 100;

	// sonu� olarak %99 luk bir basari orani g�r�yoruz.
	std::cout << "Training Accuracy : " << trainingAccuracy << std::endl;

	return 0;

}
