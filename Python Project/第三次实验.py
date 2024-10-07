
class Resolution:
    def __init__(self) -> None:
        self.__clause_set = set()

    def __init__(self, KB_input : str) -> None:
        self.__clause_set = set()
        for clause in KB_input[7 : -2].split('),('):
            #输入中前8和后2个字符是无效信息，去掉；之后按"),(""分隔，得到各子句
            if (clause[-1] == ','): self.__clause_set.add((clause[0 : -1], )) #如果子句中只有一个原子公式，就只添加它
            else:
                temp = ()
                for atom in clause.split('),'):
                    if (atom[-1] != ')'): temp = temp + ((atom + ')'), )
                    else: temp = temp + (atom, )
                self.__clause_set.add(temp)

    def ResolutionFOL(self) -> list[str]:
        #对外公开的归结方法
        clause_list = list(self.__clause_set)
        #clause_list.sort()
        count, last = len(clause_list), 0
        output = []
        for i in range(0, count): output.append(str(i + 1) + ' ' + str(clause_list[i]))
        while (clause_list[-1] != ()):
            now = len(clause_list)
            for i in range(last, len(clause_list)):
                for j in range(0, len(clause_list)):
                    collision = Resolution.__Collision(clause_list[i], clause_list[j]) #在子句中是否存在矛盾谓词
                    if (collision != ()): #若存在矛盾谓词
                        string = Resolution.__SingleStepResolution(clause_list, i, j, collision) #对这对子句单步归结
                        if (string == "F"): continue #若子句不能归结，开始下一对子句归结
                        count = count + 1 #若能归结
                        output.append(str(count) + ' ' + string) #在归结过程中添加此子句生成报告
                        if (clause_list[-1] == ()): return output #如果已生成空子句，结束归结
            last = now
        return output

    @staticmethod
    def __isVariable(term : str) -> bool:
        #判断是否是变量
        if (len(term) <= 2 and term.islower()): return True
        else: return False

    @staticmethod
    def __TermExtract(atom : str) -> list[str]:
        #将原子公式中的项拆分
        return atom[atom.rfind('(') + 1 : -1].split(',')

    @staticmethod
    def __isContradict(atom1 : str, atom2 : str) -> bool:
        #判断给定两个原子公式是否矛盾
        terms1, terms2 = Resolution.__TermExtract(atom1), Resolution.__TermExtract(atom2)
        for i in range(0, len(terms1)):
            if (Resolution.__isVariable(terms1[i]) or Resolution.__isVariable(terms2[i])): return False
            #当公式中存在变量时，不构成矛盾
        if (atom1[0] == '~' and atom2[0] != '~'):
            if (atom1[1:] == atom2): return True #当两者仅相差一个~时构成矛盾
        if (atom1[0] != '~' and atom1[0] != '~'):
            if (atom1 == atom2[1:]): return True #当两者仅相差一个~时构成矛盾
        return False

    @staticmethod
    def __Collision(clause1 : tuple[str], clause2 : tuple[str]) -> tuple[int]:
        #输入两个子句，返回冲突谓词的位置。无冲突则返回空元组
        for i in range(0, len(clause1)):
            for j in range(0, len(clause2)):
                if (clause1[i][0] != '~' and clause2[j][0] == '~'):
                    if (clause1[i][0] == clause2[j][1]): return (i, j)
                if (clause1[i][0] == '~' and clause2[j][0] != '~'):
                    if (clause1[i][1] == clause2[j][0]): return (i, j)
        return ()

    @staticmethod
    def __MGU(atom1 : str, atom2 : str) -> dict:
        #输入两个谓词，返回对第一个谓词的变量的替换
        substitution = {}
        terms1, terms2 = Resolution.__TermExtract(atom1), Resolution.__TermExtract(atom2)
        for i in range(0, len(terms1)): #找到第一个不匹配项
            if (terms1[i] != terms2[i]):
                if (Resolution.__isVariable(terms1[i]) and not Resolution.__isVariable(terms2[i])): #terms1[i]是变量而terms2[i]不是
                    substitution[terms1[i]] = terms2[i] #在代换集中添加代换terms1[i]->terms2[i]
                    atom1 = atom1.replace(terms1[i], terms2[i]) #将atom1中的变量都换为terms2[i]
                    terms1[i] = terms2[i]
                #如果都是常量或是变量则跳过
        return substitution #如果没有合法的替换，返回空字典
    
    @staticmethod
    def __Unify(clause1 : tuple[str], clause2 : tuple[str], collision : tuple[int]) -> list:
        #输入两个子句，返回两个子句的归结子句与谓词变量替换
        if (Resolution.__isContradict(clause1[collision[0]], clause2[collision[1]])):
            returner = set()
            for i in range(0, len(clause1)):
                if (i != collision[0]): returner.add(clause1[i]) #向returner中添加子句1的无冲突谓词
            for i in range(0, len(clause2)):
                if (i != collision[1]): returner.add(clause2[i]) #向returner中添加子句2的无冲突谓词
            return [tuple(returner), {}]
        substitution = Resolution.__MGU(clause1[collision[0]], clause2[collision[1]]) #寻找子句冲突谓词的变量的替换
        if (substitution == {}): return [] #如果没有合法替换，返回空列表
        atoms1 = list(clause1)
        for i in range(0, len(atoms1)):
            for var, const in substitution.items(): atoms1[i] = atoms1[i].replace(var, const) #将子句1的可换变量都替换
        returner = set()
        for i in range(0, len(atoms1)):
            if (i != collision[0]): returner.add(atoms1[i]) #向returner中添加子句1的无冲突谓词
        for i in range(0, len(clause2)):
            if (i != collision[1]): returner.add(clause2[i]) #向returner中添加子句2的无冲突谓词
        return [tuple(returner), substitution]

    @staticmethod
    def __SingleStepResolution(clause_list : list[tuple[str]], index1 : int, index2 : int, collision : tuple[int]) -> str:
        #单步归结，对可归结子句对输出生成报告，归结失败输出"F"
        clause1, clause2 = clause_list[index1], clause_list[index2] #存在矛盾谓词的一对子句
        if (len(clause2) > 1): return "F"
        unified_clause = Resolution.__Unify(clause1, clause2, collision) #由这对子句导出的归结子句及其变量代换
        if (unified_clause == [] or unified_clause[0] in clause_list): return "F" #寻找替换失败或归结子句已产生过
        clause_list.append(unified_clause[0]) #向子句集添加新的归结子句
        string = "R[" + str(index1 + 1) #生成归结子句生成报告
        if (len(clause1) != 1): string = string + chr(collision[0] + 97)
        string = string + ',' + str(index2 + 1)
        if (len(clause2) != 1): string = string + chr(collision[1] + 97)
        string = string + ']'
        if (unified_clause[1] != {}): #如果有变量替换
            string = string + '{'
            for var, const in unified_clause[1].items(): string = string + str(var) + '=' + str(const) + ','
            string = string[:-1] + '}'
        string = string + str(unified_clause[0]) #结束生成归结子句生成报告
        return string


string1 = "KB = {(GradStudent(sue),),(~GradStudent(x),Student(x)),(~Student(x),HardWorker(x)),(~HardWorker(sue),)}"
string2 = "KB = {(A(tony),),(A(mike),),(A(john),),(L(tony,rain),),(L(tony,snow),),(~A(x),S(x),C(x)),(~C(y),~L(y,rain)),(L(z,snow),~S(z)),(~L(tony,u),~L(mike,u)),(L(tony,v),L(mike,v)),(~A(w),~C(w),S(w))}"
string3 = "KB = {(On(tony,mike),),(On(mike,john),),(Green(tony),),(~Green(john),),(~On(xx,yy),~Green(xx),Green(yy))}"
test = Resolution(string3)
output = test.ResolutionFOL()
for step in output: print(step)
