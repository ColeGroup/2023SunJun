import functools

from flask import Blueprint, request, redirect, url_for, flash, render_template, session, g
import sqlite3 as sql

from werkzeug.security import generate_password_hash, check_password_hash

auth = Blueprint('auth', __name__)

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

@auth.route('/register',methods=('GET','POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        con=sql.connect('internet.db')
        db=con.cursor()
        error= None

        if not username:
            error = 'Username is required'
        elif not password:
            error ='Password is required'

        if error is None:
            try:
                db.execute(
                    "INSERT INTO user (username, password) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                con.commit()
            except con.IntegrityError:
                error=f"User {username} is already registered."
            else:
                return redirect(url_for("auth.login"))

        flash(error)
    return render_template('auth/register.html')

@auth.route('/',methods=('GET','POST'))
def login():
    fff=''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        con = sql.connect('internet.db')
        con.row_factory=dict_factory
        db = con.cursor()
        error= None
        message=''
        user = db.execute(
            'SELECT * FROM user WHERE username = ?',(username,)
        ).fetchone()
        print(user)
        if user is None:
            error ='Incorrect username.'
            message='用户名不存在'
        elif user['username']=='admin':
            error='guanli'
        elif not check_password_hash(user['password'],password):
            error='Incorrect password.'
            message='用户名或密码错误'

        if error is None:
            session.clear()
            session['user_id']=user['id']
            return redirect(url_for('index.shouye1'))

        if error=='guanli':
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('index.manage'))
        fff=message
    return render_template('auth/login.html',data=fff)

@auth.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user= None
    else:
        con = sql.connect('internet.db')
        con.row_factory = dict_factory
        db = con.cursor()
        g.user = db.execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()

@auth.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.login'))

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))
        return view(**kwargs)
    return wrapped_view

