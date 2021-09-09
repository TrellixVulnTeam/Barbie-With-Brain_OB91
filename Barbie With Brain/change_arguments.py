import pickle 

with open('args.pickle','rb') as f:
    args = pickle.load(f)
print('current args are')
print(args)

choice = input('which value would you like to change\n')

# Make sure you follow proper type
    # 'device': string 'cpu' or 'gpu'
    # 'temperature':  float less then 1
    # 'top_k':  int never change
    # 'top_p':  float less then 1
    # 'max_length' int must be larger then min_length
    # 'min_length' int must be smaller then max_length
    # 'max_history' int must be more then 1 for personality to work proerly
    # 'no_sample' bool don't change

while choice != 'end':
    if choice == 'top_k' or choice == 'min_length' or choice =='max_length' or choice == 'max_history':
        print('Must be int')
        new_value = int(input('Enter New Value'))
        args[choice]=new_value
    elif choice == 'top_p' or choice == 'temperature':
        new_value = float(input('value must be less then 1\n'))
        while new_value > 1:
            new_value = float(input('value must be less then 1\n'))
        args[choice]=new_value
    else:
        print('\n\n\nError you can only change \ntop_p \ntop_k \ntemperatur \nmin_length \nmax_length \max_jistory')
    print('\n\nNew args are')
    print(args)
    choice = input('\n\nwhich value would you like to change \nenter end to save and exit\n')


# if issues uncomment and run
# args = {
#     'device': 'cpu',
#     'temperature': 0.9,
#     'top_k': 0.0,
#     'top_p': 0.9,
#     'max_length': 20,
#     'min_length': 3,
#     'max_history': 1,
#     'no_sample': False,
# }

with open('args.pickle','wb') as f:
    pickle.dump(args,f)
print('done !')
