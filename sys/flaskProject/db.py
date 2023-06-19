import sqlite3 as sql

conn=sql.connect('internet.db')


'''conn.execute('CREATE TABLE user ('
             'id INTEGER PRIMARY KEY AUTOINCREMENT,'
             'username TEXT UNIQUE NOT NULL,'
             'password TEXT NOT NULL)')

print("table created")'''

'''conn.execute('CREATE TABLE white('
             'id INTEGER PRIMARY KEY AUTOINCREMENT,'
             'author_id INTEGER NOT NULL,'
             'w_text TEXT NOT NULL,'
             'FOREIGN KEY (author_id) REFERENCES uesr(id))')'''

'''conn.execute('CREATE TABLE black('
             'id INTEGER PRIMARY KEY AUTOINCREMENT,'
             'author_id INTEGER NOT NULL,'
             'b_text TEXT NOT NULL,'
             'FOREIGN KEY (author_id) REFERENCES uesr(id))')

print("ok")'''

print("ok")




