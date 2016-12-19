#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <algorithm>
#include <cmath>
#include <set>
#include <limits>

using namespace std;

int strToInt(string &s) {
	stringstream sin(s);
	int n;
	sin >> n;
	return n;
}

int strToMonth(const string &s) {
	stringstream sin(s);
	int year, month;
	sin >> year;
	char buf;
	sin >> buf;
	sin >> month;
	return (year - 2014) * 12 + month;
}

int minTest(int a) {
	return a != INT_MAX ? a : 0;
}

int maxTest(int a) {
	return a != INT_MIN ? a : 0;
}

void test() {
	setlocale(LC_ALL, "Russian");
	ifstream in("qiwi/e/qiwi_users_profile_data.csv");
	string s;
	getline(in, s);
	map<string, int > data;
	while (getline(in, s)) {
		stringstream sin(s);
		string user_id, sex, university, faculty, graduation_year;
		getline(sin, user_id, ';');
		getline(sin, sex, ';');
		getline(sin, university, ';');
		getline(sin, faculty, ';');
		getline(sin, graduation_year, ';');
		++data[faculty];
	}
	ifstream inn("input.txt");
	string buf;
	vector<vector<string> > dt;
	int iter = 0;
	while (getline(inn, buf)) {
		stringstream ss(buf);
		string buf2;
		dt.push_back(vector<string>());
		while (getline(ss, buf2, ',')) {
			dt.back().push_back(buf2);
		}
		++iter;
	}
	int count = 0;
	map<string, int> sp;
	for (auto i = data.begin(); i != data.end(); ++i) {
		string s;
		for (int j = 0; j < i->first.size(); ++j)
			s += tolower(i->first[j]);
		string val;
		bool flag = false;
		for (int j = 0; j < dt.size(); ++j) {
			for (int k = 1; k < dt[j].size(); ++k)
				if (s.find(dt[j][k]) != string::npos) {
					val = dt[j][0];
					flag = true;
					break;
				}
			if (flag)
				break;
		}
		if (val == "") {
			++count;
			val = "64";
		}
		sp[val] += i->second;
		//cout << s << " " << val << "\n";
	}
	//cout << count << "\n";
	for (auto i = sp.begin(); i != sp.end(); ++i)
		cout << i->first << " " << i->second << "\n";
}

struct NeuralNetwork {

	vector<int> mL;
	vector<double> mW;
	vector<double> mV;
	vector<double> mB;
	
	NeuralNetwork() {
	}
	
	NeuralNetwork(const vector<int> &pL) : mL(pL) {
		int n = 0;
		for (int i = 1; i < mL.size(); ++i)
			n += (mL[i - 1] + 1) * mL[i];
		mW.resize(n);
		int m = 0;
		for (int i = 0; i < mL.size(); ++i)
			m += mL[i];
		mV.resize(m);
		mB.resize(m);
	}
	
	void print(ostream &pOut) {
		pOut << mL.size() << '\n';
		for (int i = 0; i < mL.size(); ++i)
			pOut << mL[i] << ' ';
		pOut << '\n';
		for (int i = 0; i < mW.size(); ++i)
			pOut << mW[i] << '\n';
	}
	
	static NeuralNetwork read(istream &pIn) {
		int n;
		pIn >> n;
		vector<int> l(n);
		for (int i = 0; i < n; ++i)
			pIn >> l[i];
		NeuralNetwork nn(l);
		for (int i = 0; i < nn.mW.size(); ++i)
			pIn >> nn.mW[i];
		return nn;
	}
	
	int calc(const vector<double> &pInput) {
		for (int i = 0; i < pInput.size(); ++i)
			mV[i] = pInput[i];
		int iterW = 0;
		int iterI = 0;
		int iterO = pInput.size();
		for (int i = 1; i < mL.size(); ++i) {
			for (int j = 0; j < mL[i]; ++j) {
				double sum = 0.0;
				for (int k = 0; k < mL[i - 1]; ++k)
					sum += mV[iterI + k] * mW[iterW++];
				sum += mW[iterW++];
				mV[iterO++] = 1.0 / (1.0 + exp(-sum));
			}
			iterI += mL[i - 1];
		}
		return iterI;
	}
	
	void random() {
		for (int i = 0; i < mW.size(); ++i)
			mW[i] = (rand() / (double) RAND_MAX - 0.5) * 2.0;
	}
	
	double cost(const vector<vector<double> > &pInputs, const vector<vector<double> > &pOutputs) {
		double sum = 0.0;
		for (int i = 0; i < pInputs.size(); ++i) {
			const vector<double> &output = pOutputs[i];
			int offset = calc(pInputs[i]);
			for (int j = 0; j < output.size(); ++j) {
				double d = output[j] - mV[offset++];
				sum += d * d;
			}
		}
		return sum / pInputs.size() * 0.5;
	}
	
	void gradientDescent(const vector<double> &pV, double pR) {
		for (int i = 0; i < mW.size(); ++i)
			mW[i] += pV[i] * pR;
	}
	
	void gradientBackpropagation(const vector<vector<double> > &pInputs,
			const vector<vector<double> > &pOutputs, vector<double> &pDst) {
		for (int i = 0; i < mW.size(); ++i)
			pDst[i] = 0.0;
		for (int i = 0; i < pInputs.size(); ++i) {
			calc(pInputs[i]);
			int iterW = mW.size() - 1;
			int iterI = mB.size() - 1;
			int iterO = mB.size() - 1;
			for (int j = 0; j < mB.size(); ++j)
				mB[j] = 0.0;
			for (int j = mL[mL.size() - 1] - 1; j >= 0; --j) {
				mB[iterO] = mV[iterO] - pOutputs[i][j];
				--iterO;
			}
			for (int j = mL.size() - 1; j > 0; --j) {
				for (int k = 0; k < mL[j]; ++k) {
					mB[iterI] *= mV[iterI] * (1.0 - mV[iterI]);
					pDst[iterW--] -= mB[iterI];
					for (int p = 0; p < mL[j - 1]; ++p) {
						mB[iterO - p] += mB[iterI] * mW[iterW];
						pDst[iterW--] -= mV[iterO - p] * mB[iterI];
					}
					--iterI;
				}
				iterO -= mL[j - 1];
			}
		}
		double sum = 0.0;
		for (int i = 0; i < mW.size(); ++i)
			sum += pDst[i] * pDst[i];
		sum = 1.0 / (double) sqrt(sum);
		for (int i = 0; i < mW.size(); ++i)
			pDst[i] *= sum;
	}
};

const int PARAMETERS_COUNT = 92;

NeuralNetwork generateRandomNeuralNetwork() {
	vector<int> l;
	l.push_back(PARAMETERS_COUNT);
	l.push_back(16);
	l.push_back(1);
	NeuralNetwork nn(l);
	nn.random();
	return nn;
}

void test2() {
	
	vector<int> uu;
	vector<vector<double> > inputs, outputs;
	{
		ifstream in("output.txt");
		string s;
		getline(in, s);
		while (getline(in, s)) {
			stringstream ss(s);
			int userId;
			ss >> userId;
			char c;
			uu.push_back(userId);
			inputs.push_back(vector<double>(PARAMETERS_COUNT));
			outputs.push_back(vector<double>(1));
			for (int j = 0; j < outputs.back().size(); ++j)
				ss >> c >> outputs.back()[j];
			for (int j = 0; j < inputs.back().size(); ++j)
				ss >> c >> inputs.back()[j];
		}
		for (int j = 0; j < outputs[0].size(); ++j) {
			double sum = 0.0;
			double minValue = numeric_limits<double>::infinity();
			double maxValue = -numeric_limits<double>::infinity();
			for (int i = 0; i < outputs.size(); ++i) {
				sum += outputs[i][j];
				minValue = min(minValue, outputs[i][j]);
				maxValue = max(maxValue, outputs[i][j]);
			}
			for (int i = 0; i < outputs.size(); ++i) {
				outputs[i][j] = (outputs[i][j] - minValue) / (maxValue - minValue);
			}
		}
		for (int j = 0; j < inputs[0].size(); ++j) {
			double sum = 0.0;
			double minValue = numeric_limits<double>::infinity();
			double maxValue = -numeric_limits<double>::infinity();
			for (int i = 0; i < inputs.size(); ++i) {
				sum += inputs[i][j];
				minValue = min(minValue, inputs[i][j]);
				maxValue = max(maxValue, inputs[i][j]);
			}
			for (int i = 0; i < inputs.size(); ++i) {
				inputs[i][j] = (inputs[i][j] - minValue) / (maxValue - minValue);
			}
		}
		/*for (int i = 0; i < inputs.size(); ++i) {
			for (int j = 0; j < outputs[i].size(); ++j)
				cout << outputs[i][j] << " ";
			for (int j = 0; j < inputs[i].size(); ++j)
				cout << inputs[i][j] << " ";
			cout << "\n";
		}*/
		for (int i = 0; i < inputs.size(); ++i) {
			int id = rand() % (i + 1);
			if (id != i) {
				swap(uu[i], uu[id]);
				swap(inputs[i], inputs[id]);
				swap(outputs[i], outputs[id]);
			}
		}
	}
	
	int count = 8000;
	
	NeuralNetwork nn;
	/*{
		ifstream in("neuralnetwork.txt");
		nn = NeuralNetwork::read(in);
	}
	cout << "userId;target\n";
	for (int i = count; i < uu.size(); ++i) {
		int offset = nn.calc(inputs[i]);
		cout << uu[i] << ";" << nn.mV[offset] * (85.0 - 64.0) + 64.0 << "\n";
	}
	
	exit(0);*/
	
	nn = generateRandomNeuralNetwork();
	
	vector<double> v(nn.mW);
	{
		double k = 0.01;
		for (int i = 0; i < 1000; ++i) {
			vector<vector<double> > localInputs, localOutputs;
			int num = count / 4;
			for (int j = 0; j < num; ++j) {
				int id = rand() % inputs.size();
				localInputs.push_back(inputs[id]);
				localOutputs.push_back(outputs[id]);
			}
			/*for (int j = 0; j < count; ++j) {
				localInputs.push_back(inputs[j]);
				localOutputs.push_back(outputs[j]);
			}*/
			double prevC = nn.cost(localInputs, localOutputs);
			for (int j = 0; j < 10; ++j) {
				nn.gradientBackpropagation(localInputs, localOutputs, v);
				nn.gradientDescent(v, k);
			}
			double c = nn.cost(localInputs, localOutputs);
			if (c > prevC)
				k *= 0.5;
			cout << nn.cost(inputs, outputs) << endl;
			if (i > 0 && i % 100 == 0) {
				cout << "WARNING! FILE SAVE!" << endl;
				ofstream out("neuralnetwork.txt");
				nn.print(out);
				cout << "SAVED!" << endl;
			}
		}
	}
	exit(0);
}

void genOutput() {
	map<int, pair<double, vector<long long> > > users;
	{
		ifstream in("qiwi/e/qiwi_users_money_out_data.csv");
		string s;
		getline(in, s);
		while (getline(in, s)) {
			stringstream sin(s);
			string category, date_month, txn_count, payment, user_id;
			getline(sin, category, ';');
			getline(sin, date_month, ';');
			getline(sin, txn_count, ';');
			getline(sin, payment, ';');
			getline(sin, user_id, ';');
		}
	}
}

struct userData {
	int userId;
	string faculty;
	bool female;
	
	int outTmpLastTime;
	int outMinTransactionTime;
	int outMaxTransactionTime;
	int outMinCost;
	int outMaxCost;
	long long outSumCost;
	int outTransactionsCount;
	int outMinGap;
	int outMaxGap;
	map<string, int> outCategories;

	int inTmpLastTime;
	int inMinTransactionTime;
	int inMaxTransactionTime;
	int inMinCost;
	int inMaxCost;
	long long inSumCost;
	int inTransactionsCount;
	int inMinGap;
	int inMaxGap;
	map<string, int> inCategories;
	
	double target;
};

struct dataStruct {
	string category, date_month, txn_count, payment, user_id;
};

bool cmp(const dataStruct &a, const dataStruct &b) {
	return a.date_month < b.date_month;
}

void test3() {
	
	vector<vector<string> > dt;
	{
		ifstream inn("input.txt");
		string buf;
		while (getline(inn, buf)) {
			stringstream ss(buf);
			string buf2;
			dt.push_back(vector<string>());
			while (getline(ss, buf2, ',')) {
				dt.back().push_back(buf2);
			}
		}
	}
	
	map<int, userData> data;
	{
		ifstream in("qiwi/e/qiwi_users_profile_data.csv");
		string s;
		getline(in, s);
		while (getline(in, s)) {
			stringstream sin(s);
			string user_id, sex, university, faculty, graduation_year;
			getline(sin, user_id, ';');
			getline(sin, sex, ';');
			getline(sin, university, ';');
			getline(sin, faculty, ';');
			getline(sin, graduation_year, ';');
			userData &ud = data[strToInt(user_id)];
			ud.userId = strToInt(user_id);
			ud.faculty = faculty;
			ud.female = sex == "Жен";
			
			ud.outTmpLastTime = -1;
			ud.outMinTransactionTime = INT_MAX;
			ud.outMaxTransactionTime = INT_MIN;
			ud.outMinCost = INT_MAX;
			ud.outMaxCost = INT_MIN;
			ud.outSumCost = 0;
			ud.outTransactionsCount = 0;
			ud.outMinGap = INT_MAX;
			ud.outMaxGap = INT_MIN;
			
			ud.inTmpLastTime = -1;
			ud.inMinTransactionTime = INT_MAX;
			ud.inMaxTransactionTime = INT_MIN;
			ud.inMinCost = INT_MAX;
			ud.inMaxCost = INT_MIN;
			ud.inSumCost = 0;
			ud.inTransactionsCount = 0;
			ud.inMinGap = INT_MAX;
			ud.inMaxGap = INT_MIN;
			
			string s2;
			for (int j = 0; j < faculty.size(); ++j)
				s2 += tolower(faculty[j]);
			string val;
			bool flag = false;
			for (int j = 0; j < dt.size(); ++j) {
				for (int k = 1; k < dt[j].size(); ++k)
					if (s2.find(dt[j][k]) != string::npos) {
						val = dt[j][0];
						flag = true;
						break;
					}
				if (flag)
					break;
			}
			if (val == "") {
				val = "64";
			}
			stringstream sss(val);
			sss >> ud.target;
		}
	}
	{
		ifstream in("qiwi/e/qiwi_users_money_out_data.csv");
		string s;
		getline(in, s);
		vector<dataStruct> events;
		while (getline(in, s)) {
			stringstream sin(s);
			string category, date_month, txn_count, payment, user_id;
			getline(sin, category, ';');
			getline(sin, date_month, ';');
			getline(sin, txn_count, ';');
			getline(sin, payment, ';');
			getline(sin, user_id, ';');
			dataStruct ds;
			ds.category = category;
			ds.date_month = date_month;
			ds.txn_count = txn_count;
			ds.payment = payment;
			ds.user_id = user_id;
			events.push_back(ds);
		}
		sort(events.begin(), events.end(), cmp);
		for (int i = 0; i < events.size(); ++i) {
			string category = events[i].category;
			string date_month = events[i].date_month;
			string txn_count = events[i].txn_count;
			string payment = events[i].payment;
			string user_id = events[i].user_id;
			userData &ud = data[strToInt(user_id)];
			ud.outMinTransactionTime = min(ud.outMinTransactionTime, strToMonth(date_month));
			ud.outMaxTransactionTime = max(ud.outMaxTransactionTime, strToMonth(date_month));
			ud.outMinCost = min(ud.outMinCost, strToInt(payment));
			ud.outMaxCost = max(ud.outMaxCost, strToInt(payment));
			ud.outSumCost += strToInt(payment);
			ud.outTransactionsCount += strToInt(txn_count);
			if (ud.outTmpLastTime != -1) {
				ud.outMinGap = min(ud.outMinGap, strToMonth(date_month) - ud.outTmpLastTime);
				ud.outMaxGap = max(ud.outMaxGap, strToMonth(date_month) - ud.outTmpLastTime);
			}
			ud.outTmpLastTime = strToMonth(date_month);
			++ud.outCategories[category];
		}
	}
	{
		ifstream in("qiwi/e/qiwi_users_money_in_data.csv");
		string s;
		getline(in, s);
		vector<dataStruct> events;
		while (getline(in, s)) {
			stringstream sin(s);
			string category, date_month, txn_count, payment, user_id;
			getline(sin, category, ';');
			getline(sin, date_month, ';');
			getline(sin, txn_count, ';');
			getline(sin, payment, ';');
			getline(sin, user_id, ';');
			dataStruct ds;
			ds.category = category;
			ds.date_month = date_month;
			ds.txn_count = txn_count;
			ds.payment = payment;
			ds.user_id = user_id;
			events.push_back(ds);
		}
		sort(events.begin(), events.end(), cmp);
		for (int i = 0; i < events.size(); ++i) {
			string category = events[i].category;
			string date_month = events[i].date_month;
			string txn_count = events[i].txn_count;
			string payment = events[i].payment;
			string user_id = events[i].user_id;
			userData &ud = data[strToInt(user_id)];
			ud.inMinTransactionTime = min(ud.inMinTransactionTime, strToMonth(date_month));
			ud.inMaxTransactionTime = max(ud.inMaxTransactionTime, strToMonth(date_month));
			ud.inMinCost = min(ud.inMinCost, strToInt(payment));
			ud.inMaxCost = max(ud.inMaxCost, strToInt(payment));
			ud.inSumCost += strToInt(payment);
			ud.inTransactionsCount += strToInt(txn_count);
			if (ud.inTmpLastTime != -1) {
				ud.inMinGap = min(ud.inMinGap, strToMonth(date_month) - ud.inTmpLastTime);
				ud.inMaxGap = max(ud.inMaxGap, strToMonth(date_month) - ud.inTmpLastTime);
			}
			ud.inTmpLastTime = strToMonth(date_month);
			++ud.inCategories[category];
		}
	}
	{
		ofstream out("output.txt");
		out << "user_id;";
		
		out << "outMinTransactionTime;";
		out << "outMaxTransactionTime;";
		out << "outMinCost;";
		out << "outMaxCost;";
		out << "outSumCost;";
		out << "outTransactionsCount;";
		out << "outAvgCost;";
		out << "outMinGap;";
		out << "outMaxGap;";
		out << "outEndTime;";
		out << "outCategoriesCount;";
		
		out << "inMinTransactionTime;";
		out << "inMaxTransactionTime;";
		out << "inMinCost;";
		out << "inMaxCost;";
		out << "inSumCost;";
		out << "inTransactionsCount;";
		out << "inAvgCost;";
		out << "inMinGap;";
		out << "inMaxGap;";
		out << "inEndTime;";
		out << "inCategoriesCount\n";
		
		vector<string> outCat;
		outCat.push_back("IP-телефония");
		outCat.push_back("MCMS");
		outCat.push_back("MLM");
		outCat.push_back("QVC");
		outCat.push_back("QVP");
		outCat.push_back("QVV");
		outCat.push_back("Sim-карты для туристов");
		outCat.push_back("VPP");
		outCat.push_back("Абонентское обслуживание");
		outCat.push_back("Авиабилеты");
		outCat.push_back("Билеты в кино");
		outCat.push_back("Билеты на зрелища");
		outCat.push_back("Благотворительность");
		outCat.push_back("Бронирование гостиниц");
		outCat.push_back("Букмекеры");
		outCat.push_back("Госуслуги");
		outCat.push_back("Грузоперевозки и доставка");
		outCat.push_back("Денежные переводы");
		outCat.push_back("Ж/Д билеты");
		outCat.push_back("ЖКУ");
		outCat.push_back("Игры со ставками");
		outCat.push_back("Интернет");
		outCat.push_back("Интернет магазины");
		outCat.push_back("Информационные услуги");
		outCat.push_back("Коллекторские агентства");
		outCat.push_back("Контент");
		outCat.push_back("Купоны");
		outCat.push_back("Лотереи");
		outCat.push_back("Международная/междугородняя связь");
		outCat.push_back("Местная связь");
		outCat.push_back("Образовательные услуги");
		outCat.push_back("Онлайн игры");
		outCat.push_back("Онлайн общение");
		outCat.push_back("Оплата подписки");
		outCat.push_back("Охранные системы");
		outCat.push_back("Переводы без открытия счета по свободным реквизитам");
		outCat.push_back("Погашение кредитов");
		outCat.push_back("Предоставление займов");
		outCat.push_back("Регистрация доменов");
		outCat.push_back("Ритуальные услуги");
		outCat.push_back("Служебная");
		outCat.push_back("Создание и дизайн сайтов, трафик");
		outCat.push_back("Сотовая связь");
		outCat.push_back("Страхование");
		outCat.push_back("Такси");
		outCat.push_back("Телевидение");
		outCat.push_back("Транспортные карты");
		outCat.push_back("Туристический продукт");
		outCat.push_back("Файлообмен");
		outCat.push_back("Форекс");
		outCat.push_back("Хостинг");
		outCat.push_back("Электронные деньги");
		
		vector<string> inCat;
		inCat.push_back("QIWI");
		inCat.push_back("QIWI Казахстан");
		inCat.push_back("Агрегатор платежей");
		inCat.push_back("Банки");
		inCat.push_back("Букмекеры");
		inCat.push_back("Возвраты или корректировки платежей");
		inCat.push_back("Выплаты по рекламным акциям");
		inCat.push_back("Игры со ставками");
		inCat.push_back("Кредиты и займы на кошелек");
		inCat.push_back("Кэшбэк МегаФон");
		inCat.push_back("Лотереи");
		inCat.push_back("Мобильная коммерция");
		inCat.push_back("Онлайн игры");
		inCat.push_back("Остальное");
		inCat.push_back("Терминалы и отделения партнеров QIWI");
		inCat.push_back("Форекс");
		inCat.push_back("Электронные деньги");
		
		for (auto i = data.begin(); i != data.end(); ++i) {
			out << i->first << ";";
			
			out << i->second.target << ";";
			
			out << minTest(i->second.outMinTransactionTime) << ";";
			out << maxTest(i->second.outMaxTransactionTime) << ";";
			out << minTest(i->second.outMinCost) << ";";
			out << maxTest(i->second.outMaxCost) << ";";
			out << i->second.outSumCost << ";";
			out << i->second.outTransactionsCount << ";";
			out << (i->second.outTransactionsCount != 0 ?
					i->second.outSumCost / (double) i->second.outTransactionsCount : 0) << ";";
			out << (i->second.outMinGap != INT_MAX ? i->second.outMinGap : 0) << ";";
			out << (i->second.outMaxGap != INT_MIN ? i->second.outMaxGap : 0) << ";";
			out << strToMonth("2017-1") - maxTest(i->second.outMaxTransactionTime) << ";";
			out << i->second.outCategories.size() << ";";
			for (int j = 0; j < outCat.size(); ++j)
				out << i->second.outCategories[outCat[j]] << ";";
			
			out << minTest(i->second.inMinTransactionTime) << ";";
			out << maxTest(i->second.inMaxTransactionTime) << ";";
			out << minTest(i->second.inMinCost) << ";";
			out << maxTest(i->second.inMaxCost) << ";";
			out << i->second.inSumCost << ";";
			out << i->second.inTransactionsCount << ";";
			out << (i->second.inTransactionsCount != 0 ?
					i->second.inSumCost / (double) i->second.inTransactionsCount : 0) << ";";
			out << (i->second.inMinGap != INT_MAX ? i->second.inMinGap : 0) << ";";
			out << (i->second.inMaxGap != INT_MIN ? i->second.inMaxGap : 0) << ";";
			out << strToMonth("2017-1") - maxTest(i->second.inMaxTransactionTime) << ";";
			out << i->second.inCategories.size() << ";";
			for (int j = 0; j < inCat.size(); ++j)
				out << i->second.inCategories[inCat[j]] << ";";
			
			out << i->second.female << "\n";
		}
	}
	exit(0);
}

struct dsu {
	dsu *p;
	dsu() : p(0) {}
};

dsu *root(dsu *a) {
	return a->p == 0 ? a : (a->p = root(a->p));
}

void merge(dsu *a, dsu *b) {
	a = root(a);
	b = root(b);
	b->p = a;
}

void test4() {
	{
		ifstream in("output.txt");
		string s;
		getline(in, s);
		vector<int> vId;
		vector<vector<double> > v;
		int count = 0;
		while (getline(in, s)) {
			stringstream ss(s);
			string buf;
			getline(ss, buf, ';');
			vId.push_back(strToInt(buf));
			char c;
			double val;
			v.push_back(vector<double>());
			while (ss >> val) {
				v.back().push_back(val);
				if (!(ss >> c))
					break;
			}
			if (v.back()[5] + v.back()[16] <= 10.5) {
				v.pop_back();
				vId.pop_back();
			}
		}
		for (int j = 0; j < v[0].size(); ++j) {
			double sum = 0.0;
			double minValue = numeric_limits<double>::infinity();
			double maxValue = -numeric_limits<double>::infinity();
			for (int i = 0; i < v.size(); ++i) {
				sum += v[i][j];
				minValue = min(minValue, v[i][j]);
				maxValue = max(maxValue, v[i][j]);
			}
			for (int i = 0; i < v.size(); ++i) {
				v[i][j] = (v[i][j] - sum / v.size()) / (maxValue - minValue);
			}
		}
		vector<pair<double, pair<int, int> > > lens;
		for (int i = 0; i < v.size(); ++i) {
			for (int j = 0; j < i; ++j) {
				double sum = 0.0;
				for (int k = 0; k < v[0].size(); ++k)
					sum += (v[i][k] - v[j][k]) * (v[i][k] - v[j][k]);
				lens.push_back(make_pair(sqrt(sum), make_pair(i, j)));
			}
			cout << i << endl;
		}
		sort(lens.begin(), lens.end());
		vector<dsu> d(v.size());
		int components = v.size();
		ofstream test_stream("test.txt");
		int iter = 1;
		for (int i = 0; i < lens.size(); ++i) {
			dsu *a = &d[lens[i].second.first];
			dsu *b = &d[lens[i].second.second];
			if (root(a) != root(b)) {
				merge(a, b);
				--components;
				map<dsu*, int> mp;
				for (int j = 0; j < v.size(); ++j)
					++mp[root(&d[j])];
				test_stream << components << " " << lens[i].first << " ";
				for (auto j = mp.begin(); j != mp.end(); ++j)
					if (j->second >= 10)
						test_stream << j->second << " ";
				test_stream << endl;
				cout << i << endl;
				if (iter == 3466) {
					test_stream << "start\n";
					map<dsu*, vector<int> > mp2;
					for (int j = 0; j < v.size(); ++j)
						mp2[root(&d[j])].push_back(vId[j]);
					for (auto j = mp2.begin(); j != mp2.end(); ++j) {
						vector<int> &dd = j->second;
						for (int k = 0; k < dd.size(); ++k)
							test_stream << dd[k] << " ";
						test_stream << "\n";
					}
					exit(0);
				}
				++iter;
			}
		}
		//for (int i = 0; i < v.size(); ++i) {
		//	for (int j = 0; j < v[i].size(); ++j)
		//		cout << v[i][j] << " ";
		//	cout << endl;
		//}
	}
	exit(0);
}

int main() {
	/*set<int> vId;
	
	{
		ifstream in("groups.txt");
		string s;
		getline(in, s);
		getline(in, s);
		stringstream ss(s);
		int id;
		while (ss >> id) {
			vId.insert(id);
		}
	}
	
	{
		ifstream in("output.txt");
		string s;
		getline(in, s);
		vector<vector<double> > v;
		int count = 0;
		while (getline(in, s)) {
			stringstream ss(s);
			string buf;
			getline(ss, buf, ';');
			char c;
			double val;
			v.push_back(vector<double>());
			while (ss >> val) {
				v.back().push_back(val);
				if (!(ss >> c))
					break;
			}
			if (vId.find(strToInt(buf)) == vId.end()) {
				v.pop_back();
			}
		}
		for (int i = 0; i < v.size(); ++i) {
			for (int j = 0; j < v[i].size(); ++j)
				cout << v[i][j] << " ";
			cout << endl;
		}
	}
	
	return 0;*/
	
//	test3();
	test2();
	test4();
	
	genOutput();
	
	test2();
	
	ifstream in("qiwi/e/qiwi_users_money_out_data.csv");
	
	string s;
	getline(in, s);
	
	map<string, map<int, int> > categories;
	map<int, int> hero;
	while (getline(in, s)) {
		stringstream sin(s);
		string category, date_month, txn_count, payment, user_id;
		getline(sin, category, ';');
		getline(sin, date_month, ';');
		getline(sin, txn_count, ';');
		getline(sin, payment, ';');
		getline(sin, user_id, ';');
		++hero[strToInt(payment)];
		++categories[category][strToInt(payment)];
	}
	
	ofstream html("index.html");
	
	html << R"HERO(<!DOCTYPE html>
<html>
	<head>
		<title>Test</title>
		<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
		<script src="jquery-1.11.2.min.js"></script>
		<style type="text/css">
			* {
				margin: 0px;
				padding: 0px;
				border-spacing: 0px;
			}
		</style>
	</head>
	<body>
		<canvas id="canvas" width=0 height=0>
		</canvas>
		<script>
			function vt(x, y) {
				this.x = x;
				this.y = y;
			}
		
			var elem = document.getElementById("canvas");
			var ctx = elem.getContext("2d");
			var size = new vt(10000, 10000);
			elem.width = size.x;
			elem.height = size.y;
			
			var data = [)HERO";
	
	bool flag = false;
	for (auto j = categories.begin(); j != categories.end(); ++j) {
		map<int, int> &d = j->second;
		if (flag)
			html << ",";
		cout << j->first << " " << d.size() << "\n";
		html << "{c:" << "'test'" << ",d:[";
		bool flag2 = false;
		for (auto i = d.begin(); i != d.end(); ++i) {
			if (flag2)
				html << ",";
			html << "{x:" << i->first << ",y:" << i->second << "}";
			flag2 = true;
		}
		html << "]}";
		flag = true;
	}
	
html << R"HERO(];
			
			function render(id) {
				ctx.clearRect(0, 0, size.x, size.y);
				var d = data[id].d;
				for (var i = 0; i < d.length; ++i) {
					ctx.beginPath();
					ctx.rect(d[i].x / 10000 * size.x, size.y, 10, -d[i].y / 2000 * size.y);
					ctx.fillStyle = "#ff0000";
					ctx.fill();
				}
			}
			render(0);
		</script>
	</body>
</html>)HERO";
	
	return 0;
}