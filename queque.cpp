#include<iostream>
#include<stdexcept>
using namespace std;
template<typename T>
struct Node
{
    T data;
    Node*next;
    Node(const T& value,Node*n=nullptr):data(value),next(n){}
};
template<typename T>
struct queue
{
    private:
    Node<T>*rear;
    Node<T>*front;
    int count;
    public:
    queue():rear(nullptr),front(nullptr),count(0){}
    ~queue(){
        while(!isEmpty()){
            dequeue();
        }
    }
    bool isEmpty(){
        return front==nullptr;
    }
    void dequeue(){
        if(isEmpty(){
            return;
        })
        Node<T> *temp=front;
        front=front->next;
        delete temp;
        if(front=-nullptr){
            rear=front;
        }
        count--;
    }
    void enqueue(const T& value){
        Node<T> *temp=new Node<T>(value);
        if(isEmpty){
            front=rear=temp;
            count++;
            return;
        }
        rear->next=temp;
        rear=temp;
        count++;
    }
    int size(){
        return count;
    }
    T frontValue(){
        if(front==nullptr){
            return;
        }
        return front->data;
    }
    
};


