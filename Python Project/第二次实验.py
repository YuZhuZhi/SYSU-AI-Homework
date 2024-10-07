
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class StuData:
    def __init__(self, filename: str) -> None:
        self.data = []
        with open(filename) as file_obj:
            for line in file_obj.readlines():
                self.data.append(line.split())
                self.data[-1][3] = int(self.data[-1][3])

    def AddData(self, **single_stu) -> None:
        self.data.append([ single_stu['name'], single_stu['stu_num'], single_stu['gender'], single_stu['age'] ])

    def SortData(self, status: str) -> None:
        map = {'name' : 0, 'stu_num' : 1, 'gender' : 2, 'age' : 3}
        self.data.sort(key = lambda statuss : statuss[map[status]])

    def ExportFile(self, filename: str) -> None:
        with open(filename, 'w') as file_obj:
            for single_stu in self.data:
                file_obj.write(" ".join([str(status) for status in single_stu]) + '\n')


students = StuData("student_data.txt")
print(students.data)
students.AddData(name = "Bob", stu_num = "003", gender = "M", age = 20)
print(students.data)
students.SortData('stu_num')
print(students.data)
students.ExportFile("output.txt")
