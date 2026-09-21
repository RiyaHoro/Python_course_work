from abc import ABC, abstractmethod


class Person(ABC):

    def __init__(self, name, age, gender, address):
        self.name = name
        self.age = age
        self.gender = gender
        self.address = address

    def __str__(self):
        return f"Name: {self.name}, Age: {self.age}, Gender: {self.gender}, Address: {self.address}"

    def greet(self, other_person):
        print(f"Hello {other_person.name}! My name is {self.name}.")

    @abstractmethod
    def introduce(self):
        pass

    @staticmethod
    def is_adult(age):
        return age >= 18


class Student(Person):

    def __init__(self, name, age, gender, address):
        super().__init__(name, age, gender, address)

    def introduce(self):
        print(f"Hi, my name is {self.name}.")


# Example
p1 = Student("Riya", 22, "Female", "Ranchi")
p2 = Student("Ananya", 20, "Female", "Bokaro")

print(p1)
p1.greet(p2)
p1.introduce()

print(Person.is_adult(22))
print(Person.is_adult(16))