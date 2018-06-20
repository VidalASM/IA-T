#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>

using namespace std;

const int CANTIDAD_ENTRADAS = 2;
const int CANTIDAD_SALIDAS = 1;
const int CANTIDAD_CAPAS = 1;
const int CAPAS[] = {3};
const int EPOCAS = 3;
const double TASA_APRENDIZAJE = 0.5;
const int PUNTOS_ENTRENAMIENTO = 4;

double sigmoidal(double x){
	return 1.0/(1+exp(-x));
}

double sigmoidalp(double x){
	double s =  sigmoidal(x);
	return s*(1-s);
}

class Neurona{
public:
	int id;
	double dato[2];
	static int cont_id;
	virtual double salida() = 0;
	virtual bool calculado() = 0;
	virtual void reiniciar() = 0;
	virtual void propagarError() = 0;
	virtual void recibeGradiente(double) = 0;
	virtual void muestraTopologia() = 0;
};
int Neurona::cont_id = 0;

class NeuronaEntrada: public Neurona{
public:
	NeuronaEntrada(){
		this->id = Neurona::cont_id ++;
	}

	virtual double salida(){ return this->dato[0] + this->dato[1]; }

	virtual bool calculado(){ return true; }

	virtual void reiniciar(){}

	virtual void propagarError(){}

	virtual void recibeGradiente(double g){}

	virtual void muestraTopologia(){
		cout<<"<Entrada id="<<this->id<<">"<<endl;
	}
};

class Dendrita{
public:
	Neurona* neurona;
	double peso;
	Dendrita(Neurona*& n, double p){
		this->neurona = n;
		this->peso = p;
	}
};

class NeuronaOculta :public NeuronaEntrada{
public:
	double bias;
	vector<Dendrita> entradas;
	double gradiente;
	double calculo;
	double activacion;

	void agregarEntrada(Neurona* n, double peso=0){
		entradas.push_back(Dendrita(n, peso));
	}

	virtual double salida(){
		calculo = bias;
		for(Dendrita& d: entradas){
			calculo += d.peso * (d.neurona->dato[0] + d.neurona->dato[1]);
		}
		activacion = sigmoidal(calculo);
		return calculo;
	}

	virtual void reiniciar(){
		activacion = 0.0;
		gradiente = 0.0;
	}

	virtual void recibeGradiente(double g){
		this->gradiente += g;
	}

	virtual void propagarError(){
		for(unsigned i = 0; i < entradas.size(); i++){
			entradas[i].neurona->recibeGradiente(this -> gradiente * entradas[i].peso);
		}
	}

	virtual void ajustaPesos(double etha){
		this->gradiente *= sigmoidalp(this->calculo);
		double dw = etha * this -> gradiente * this -> activacion;
		cout<<"Ajusta pesos NO "<<etha<<", "<<this->gradiente<<", "<<this->activacion<<", "<<dw<<endl;
		for(Dendrita& d: entradas){
			d.peso += dw;
		}
	}

	virtual void muestraTopologia(){
		cout<<"Oculta id="<<this->id<<" entradas={ ";
		for(unsigned i=0;i<entradas.size();i++){
			cout<<"("<<entradas[i].neurona->dato[0]<<" "<<entradas[i].neurona->dato[1]<<", "<<entradas[i].peso<<") ";
		}
		cout<<"}"<<endl;
	}

};

class NeuronaSalida: public NeuronaOculta{
public:
	virtual void ajustaError(double y){
		this -> gradiente = (this->activacion - y);
	}

	virtual void ajustaPesos(double etha){
		double dw = etha * this -> gradiente * this -> activacion;
		cout<<"Ajusta pesos "<<etha<<", "<<this->gradiente<<", "<<this->activacion<<", "<<dw<<endl;
		for(Dendrita& d: entradas){
			d.peso += dw;
		}
	}

	virtual void muestraTopologia(){
		cout<<"<Salida id="<<this->id<<" entradas={ ";
		for(unsigned i=0;i<entradas.size();i++){
			cout<<"("<<entradas[i].neurona->dato[0]<<" "<<entradas[i].neurona->dato[1]<<", "<<entradas[i].peso<<") ";
		}
		cout<<"}>"<<endl;
	}

	virtual double salida(){
		calculo = bias;
		for(Dendrita& d: entradas){
			calculo+=d.peso * (d.neurona->dato[0] + d.neurona->dato[1]);
		}
		return activacion = calculo;
	}

	virtual void propagarError(){
		for(unsigned i = 0; i < entradas.size(); i++){
			entradas[i].neurona->recibeGradiente(this -> gradiente * entradas[i].peso);
		}
	}

};

class RedNeuronal{
public:
	vector<NeuronaEntrada*> neuronasEntrada;
	vector<NeuronaOculta*> neuronasIntermedias;
	vector<NeuronaSalida*> neuronasSalida;

	double etha;
	double aprendizaje;
	double error;
	double errorPrevio;

	RedNeuronal(int entradas, int salidas, int capasIntermedias, const int cantidades[], double e){
		etha = e;
		aprendizaje = e;
		for(int i=0;i<entradas;i++){
			neuronasEntrada.push_back(new NeuronaEntrada());
		}
		for(int i=0;i<cantidades[0];i++){
			NeuronaOculta* n = new NeuronaOculta();
			n->bias = 1.0;
			for(int j=0;j<entradas;j++){
				n->agregarEntrada(neuronasEntrada[j], 0.0);
			}
			neuronasIntermedias.push_back(n);
		}
		int inicioCapa = 0;
		for(int i=1;i<capasIntermedias;i++){
			for(int j=0;j<cantidades[i];j++){
				NeuronaOculta* n = new NeuronaOculta();
				n->bias = 1.0;
				for(int k = 0; k<cantidades[i-1]; k++){
					n->agregarEntrada(neuronasIntermedias[inicioCapa + k], 0.0);
				}
				neuronasIntermedias.push_back(n);
			}
			inicioCapa += cantidades[i-1];
		}
		for(int i=0;i<salidas;i++){
			NeuronaSalida* n = new NeuronaSalida();
			n->bias = 1.0;
			for(unsigned j = inicioCapa; j<neuronasIntermedias.size(); j++){
				n->agregarEntrada(neuronasIntermedias[j], 0.0);
			}
			neuronasSalida.push_back(n);
		}
	}

	void reiniciarNeuronas(){
		for(NeuronaSalida*& n: neuronasSalida){
			n->reiniciar();		
		}
		for(NeuronaOculta*& n: neuronasIntermedias){
			n->reiniciar();
		}
	}

	void inicializaError(){
		errorPrevio = error;
		error = 0;
	}

	double evalua(double* input){
		neuronasEntrada[0]->dato[0] = input[0];
		neuronasEntrada[0]->dato[1] = input[1];
		reiniciarNeuronas();
		return neuronasSalida[0]->salida();
	}

	void entrena(double* input, double output){
		double y = evalua(input);
		cout<<(output - y)<<endl;
		error += 0.5 * (output - y) * (output - y);
		neuronasSalida[0]->ajustaError(output);
		if (output - y != 0.0){
			neuronasSalida[0]->ajustaPesos(aprendizaje);
		}		
		neuronasSalida[0]->propagarError();
		for(int i = neuronasIntermedias.size() - 1; i >= 0 ; --i){
			if (output - y != 0.0){
				neuronasIntermedias[i]->ajustaPesos(aprendizaje);
			}
			neuronasIntermedias[i]->propagarError();
		}
	}

	void ajustaAprendizaje(){
		if(error > errorPrevio){	
			aprendizaje *= etha;
		}
	}

	double errorMedio(){
		return error;
	}

	void muestraTopologia(){
		for(NeuronaSalida* n: neuronasSalida){
			n->muestraTopologia();		
		}
		for(NeuronaOculta* n: neuronasIntermedias){
			n->muestraTopologia();
		}
		for(Neurona* n: neuronasEntrada){
			n->muestraTopologia();
		}
	}
};

int main(){
	cout<<"Estado de prueba"<<endl;
	RedNeuronal rna(CANTIDAD_ENTRADAS, CANTIDAD_SALIDAS, CANTIDAD_CAPAS, CAPAS, TASA_APRENDIZAJE);
	double entradas[PUNTOS_ENTRENAMIENTO][2] = {{0.0,0.0},{0.0,1.0},{1.0,0.0},{1.0,1.0}};
	double salidas[PUNTOS_ENTRENAMIENTO] = {0.0, 1.0, 1.0, 0.0};

	cout<<"Calculados los datos de entrenamiento: "<<PUNTOS_ENTRENAMIENTO<<endl;
	cout<<"Inicia el entrenamiento"<<endl;
	for(int k = 0; k < EPOCAS; k++){
		rna.inicializaError();
		for(int i = 0; i < PUNTOS_ENTRENAMIENTO ; i++){
			rna.entrena(entradas[i], salidas[i]);
		}
		cout<<"\nEpoca: "<<k+1<<"/"<<EPOCAS<<". Error: "<<rna.errorMedio()<<endl;
		rna.ajustaAprendizaje();
		rna.muestraTopologia();
	}
	rna.muestraTopologia();

	return 0;
}