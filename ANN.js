//функции вывода
function printMatrix(matrix, txt) { // визуализация... модифицировать и ВИЗУАЛИЗИРОВАТЬ
	if (txt) document.body.appendChild(document.createElement("p")).innerHTML = txt;
	for (var i = 0; i < matrix.length; i++) {
		var str = "";
		for (var j = 0; j < matrix[0].length; j++)
			str += matrix[i][j] + " ";
		document.body.appendChild(document.createElement("p")).innerHTML = str;
	}
	document.body.appendChild(document.createElement("hr"));
}
function print(x) {
	document.body.appendChild(document.createElement("p")).innerHTML = x;
	document.body.appendChild(document.createElement("hr"));
}
//прочие функции
function activationFunc(x) { // функция активации
	for (var i = 0; i < x.length; i++)
		for (var j = 0; j < x[0].length; j++)
			Math.tanh(x[i][j]);
			//x[i][j] = 1/(1+Math.pow(Math.E, -x[i][j])); // сигмоида
	return x;
}
function dActivFunc(x) {
	for (var i = 0; i < x.length; i++)
		for (var j = 0; j < x[0].length; j++)
			x[i][j] = 1 / Math.cosh(x[i][j]); // гиперболический тангенс
			//x[i][j] = activationFunc(x[i][j])*(1-activationFunc(x[i][j])); // сигмоида
	return x;
}
function objLength(obj) {
	var length = 0;
	for (var key in obj) length++;
	return length;
}
//работа с матрицами
function matrixCreate(rows, columns){ // создание матрицы
	var arr = [];
	for(var i=0; i<rows; i++){
		arr[i] = [];
		for(var j=0; j<columns; j++)
			arr[i][j] = i+j;
	}
	return arr;
}
function matrixTranspose(matrix) { // транспонирование
	var transposed = matrixCreate(matrix[0].length, matrix.length);
	for (var i = 0; i < matrix.length; i++)
		for (var j = 0; j < matrix[0].length; j++)
			transposed[j][i] = matrix[i][j];
	return transposed;
}
function matrixMultiplication(A, B) { // скалярное произведение матрицы
	function arrSum(arr) {
		var sum = 0;
		for (var i = 0; i < arr.length; i++)
			sum += arr[i];
		return +sum.toFixed(10);
	}
	var C = matrixCreate(A.length, B[0].length);
	for (var i = 0; i < A.length; i++) { // строки
		for (var j = 0; j < B[0].length; j++) {	// столбцы
			var arrTmp = []; 
			for (var m = 0; m < A[0].length; m++) {
				arrTmp[m] = A[i][m]*B[m][j];
			}
			C[i][j] = arrSum(arrTmp);
		}
	}
	return C;
}
// функции нейронной сети
function Inn(layers, trainSet) { // инициализация нейронной сети
	function gauss(x) {
		var g = 1/(Math.sqrt(0.2)*2.5)*Math.exp(-(Math.pow((x), 2)/(2*0.2)));
		if (x >= 0) return g;
		return -g;
	}
	function normalDistribution() {
		return +(gauss((Math.random()*2)-1)).toFixed(10);
	}
	
	for (var i = 1 ; i < layers.length; i++) {
		matrix = matrixCreate(layers[i], layers[i-1]);
		for (var m = 0; m < matrix.length; m++)
			for (var l = 0; l < matrix[m].length; l++)
				matrix[m][l] = normalDistribution();
		this[i] = matrix;
	}
	// вкючаем в нейронку первый набор обучения
	this[0] = [];
	for (var key in trainSet) {
		for (var j = 0; j < trainSet[key][0].length; j++)
			this[0].push([trainSet[key][0][j]]);
		this[objLength(this)] = [trainSet[key][1]];
		break;
	}
}
function Query(layers) { // конструктор объекта ответов нейронной сети
	var previousLayer = layers[0];
	var length = objLength(layers)-1;
	this[0] = previousLayer;
	for (var i = 1; i < length; i++)
		this[i] = previousLayer = activationFunc(matrixMultiplication(layers[i], previousLayer));
}
function train(trainSet) { // старт одной эпохи обучения
	function ErrorCalculating(layers, out) {
		function outputErrorCalculating(expectedValue, realValue) { // ошибка на выходе
			C = matrixCreate(realValue.length, realValue[0].length);
			for (var i = 0; i < realValue.length; i++)
				for (var j = 0; j < realValue[0].length; j++)
					C[i][j] = +(expectedValue[i][j]-realValue[i][j]).toFixed(10);
			return C;
		}
		function hiddenErrorCalculating(layerWeightsOriginal, errors) { // ошибка скрытого слоя. Очень сильно нагружает при объёмных расчётах, но даёт более успешные результаты обучения
			// создаём копию оригинала layerWeightsOriginal принудительно
			var layerWeights = matrixCreate(layerWeightsOriginal.length, layerWeightsOriginal[0].length);
			// заполнить копию
			for (var i = 0; i < layerWeights.length; i++)
				for (var j = 0; j < layerWeights[0].length; j++)
					layerWeights[i][j] = layerWeightsOriginal[i][j];
			// сумма по вертикали
			var verticalSum = new Array();
			for (var i = 0; i < layerWeights.length; i++)
				verticalSum[i] = 0;
			for (var i = 0; i < layerWeights.length; i++)
				for (var j = 0; j < layerWeights[i].length; j++) {
					verticalSum[i] += layerWeights[i][j];
					verticalSum[i] = +verticalSum[i].toFixed(10);
				}
			// деление значения i,j-того транспонированной матрицы на j значение из массива сумм
			layerWeights = matrixTranspose(layerWeights);
			for (var i = 0; i < layerWeights.length; i++)
				for (var j = 0; j < layerWeights[i].length; j++) {
					layerWeights[i][j] /= verticalSum[j];
					layerWeights[i][j] = +layerWeights[i][j].toFixed(10);
				}
			return matrixMultiplication(layerWeights, errors);
		}
		var length = objLength(layers)-1;
		var error = [];
		error[length] = outputErrorCalculating(layers[length], out); // ошибка на выходе
		this[length] = error[length];
		for (var i = length-1; i > 0; i--) {
			error[i] = matrixMultiplication(matrixTranspose(layers[i]), error[i+1]);   // ТУТ ВАРИАЦИЯ - полный расчёт или быстрый
			this[i] = error[i];
		}
	}
	function deltaCalculating (j, error, out) {
		var errorsMatrix = matrixCreate(error[j+1].length, 1);
		//высчитываем все значения из следующего) слоя
		//var sigm = dActivFunc(activationFunc(matrixMultiplication(layers[j], out[j-1])));
		var th = dActivFunc(matrixMultiplication(layers[j], out[j-1]));
		for (var i = 0; i < errorsMatrix.length; i++) {
			errorsMatrix[i][0] = -2*error[j+1][i]*th[i][0]/*sigm[i][0]*/*speed;
		}
		//вернуть матрицу с высчитанными дельтами
		return matrixMultiplication(errorsMatrix, matrixTranspose(out[j-1]));
	}
	//готовим массив со всеми ошибками для возврата
	var returnErrors = [];
	// входные данные генерируем внутри, в зависимости от trainSet'а
	for (var key in trainSet) {
		layers[0] = layers[objLength(layers)-1] = []; // обнуляем вход и ожидание
		for (var i = 0; i < trainSet[key][0].length; i++)
			layers[0].push([trainSet[key][0][i]]); // входные данные
		layers[objLength(layers)-1] = [trainSet[key][1]]; //подгружаем ошибки
		var queryObj = new Query(layers); // сохранить результат вызова query в отдельный объект
		var errors = new ErrorCalculating(layers, queryObj[objLength(queryObj)-1]); // получить ошибки для каждого слоя
		returnErrors.push(errors[objLength(layers)-1])
		var deltas = {};
		for (var i = 1; i < objLength(layers)-1; i++)
			deltas[i] = deltaCalculating(i, errors, queryObj); //высчитать дельты изменений весов
		//считаем новые значения синапсов
		for (var i = 1; i < objLength(layers)-1; i++)
			for (var j = 0; j < layers[i].length; j++)
				for (var k = 0; k < layers[i][j].length; k++)
					layers[i][j][k] = +(layers[i][j][k]-deltas[i][j][k]).toFixed(10);
	}
	return returnErrors;
}
function start() {
	print("<pre>Нейронная сеть инициализирована с такими параметрами</pre>");
	for (var i = 1; i < objLength(layers)-1; i++)
		printMatrix(layers[i], "layer"+i);

	print("<pre>"+numOfEpochs+" эпох<br>скорость обучения = "+speed+"</pre>");
	for (var i = 0; i < numOfEpochs; i++)
		var tr = train(trainSet);
	
	for (var i = 1; i < objLength(layers)-1; i++)
		printMatrix(layers[i], "layer"+i);
	
	for (var key in trainSet) {
		layers[0] = [];
		for (var j = 0; j < trainSet[key][0].length; j++) {
			layers[0].push([trainSet[key][0][j]]);
		}
		layers[objLength(layers)-1] = [trainSet[key][1]];
		answer = new Query(layers);
		printMatrix(answer[objLength(answer)-1], "ожидая"+trainSet[key][1][0]);
	}
}
// инициализируем нейронную сеть
var layers = new Inn(layers, trainSet);