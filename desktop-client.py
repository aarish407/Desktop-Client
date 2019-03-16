import tkinter as tk
import sqlite3 as sq

def main():   

    root= tk.Tk()
    root.title("Desktop-Client")
    root.geometry("600x600")
   
    ######### IP ADDRESS OF SERVER ################

    ip_frame= tk.Frame()
    ip= tk.StringVar(ip_frame)
    ip_label = tk.Label(ip_frame, text="Server IP Address")
    ip_entry= tk.Entry(ip_frame, textvariable= ip)
    ip_entry.pack(padx= 5, pady= 5, side=tk.RIGHT)
    ip_label.pack(padx= 5, pady= 5, side=tk.LEFT)
    ip_frame.pack()
     
    ########## USERNAME ################

    username_frame= tk.Frame()
    username= tk.StringVar(username_frame)
    username_label = tk.Label(username_frame, text="User Name")
    username_entry= tk.Entry(username_frame, textvariable= username)
    username_entry.pack(padx= 5, pady= 5, side=tk.RIGHT)
    username_label.pack(padx= 5, pady= 5, side=tk.LEFT)
    username_frame.pack()

    ########## PASSWORD ##################

    password_frame= tk.Frame()
    password= tk.StringVar(password_frame)
    password_label = tk.Label(password_frame, text="Password")
    password_entry= tk.Entry(password_frame, show= "*", textvariable= password)
    # password_entry.bind('<Return>',get_password)
    password_entry.pack(padx= 5, pady= 5, side= tk.RIGHT)
    password_label.pack(padx= 5, pady= 5, side= tk.LEFT)
    password_frame.pack()

    root.mainloop()

    print("This username ",username.get(), "This password ",password.get(), "IP Address ",ip.get())

if __name__ == '__main__':
    main()