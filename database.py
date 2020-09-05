from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, DateTime
from sqlalchemy import Index, create_engine
from sqlalchemy.orm import relationship, backref, sessionmaker

import logging
import settings

Base = declarative_base()

# specify database configurations
config = settings.config
db_user = settings.config.get('user')
db_pwd = config.get('password')
db_host = config.get('host')
db_port = config.get('port')
db_name = config.get('database')
# specify connection string
connection_str = f'mysql+pymysql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}'

engine = create_engine(connection_str, echo=True)

Session = sessionmaker(bind=engine)
session = Session()

# dirty_


# id = session.query(Restaurant).filter(Restaurant.restaurant_id == val)
# id = session.query(id.exists()).scalar()


# class SteamApps(Base):
#     __tablename__ = 'steamapps'
#     steam_appid = Column(String(250), index=True, primary_key=True)
#     last_update = Column(DateTime)
# # categories = relationship('Category', secondary = 'restaurant_category')
#
#
# class DirtyData(Base):
#     __tablename__ = 'dirty_data'
#     steam_appid = Column(Integer, primary_key=True)
#     type = Column(String(10))
#     game_name = Column(String(250))
#     required_age = Column(Integer)
#     is_free = Column(Boolean)
#     short_description = Column(String(250))
#     supported_languages = Column(String(250))  # todo: type? length?
#     website = Column(String(250))
#     developers = Column(String(250))  # todo: could map to another table to easily look at patterns
#     publishers = Column(String(250))  # todo: could map to another table to easily look at patterns
#     packages = Column(String(250))  # todo: type? length?
#     package_groups = Column(String(250))  # todo: type? length?
#     controller_support = Column(Boolean)
#     drm_notice = Column(String(250))  # todo: type? length?
#
#     ext_user_account_notice = Column(String(250))  # todo: type? length?
#
#     coming_soon = Column(Boolean)
#
#     release_date = Column(DateTime)
#     price_currency = Column(String(250))
#     price_discount_percent = Column(Float)
#     price_final = Column(Float)
#     price_initial = Column(Float)
#     price_recurring_sub = Column(Float)
#     price_recurring_sub_desc = Column(String(250))  # todo: length?
#
#     windows_support = Column(Boolean)
#     mac_support = Column(Boolean)
#     linux_support = Column(Boolean)
#
#     recommendations = Column(Float)
#     metacritic_score = Column(Float)
#     # fullgame_appid todo: how does this map back onto itself?
#     achievement_count = Column(Float)
#     # demo_appid  todo: how does this map back onto itself?
#     screenshot_count = Column(Float)
#     movie_count = Column(Float)
#     dlc_count = Column(Float)
#
#     categories = relationship('Category', secondary='app_category')
#     genres = relationship('Genre', secondary='app_genre')
#
#
# class Category(Base):
#     __tablename__ = 'category'
#     category_id = Column(Integer, primary_key=True)
#     apps = relationship('dirty_data', secondary='app_category')
#     name = Column(String(250), nullable=False)
#
#
# class App_Category(Base):
#     __tablename__ = 'app_category'
#     category_id = Column(Integer, ForeignKey('category.category_id'),
#                          primary_key=True)
#     steam_appid = Column(String(250), ForeignKey('dirty_data.steam_appid'),
#                          primary_key=True)
#
#
# class Genre(Base):
#     __tablename__ = 'gerne'
#     genre_id = Column(Integer, primary_key=True)
#     apps = relationship('dirty_data', secondary='app_category')
#     name = Column(String(250), nullable=False)
#
#
# class App_Genre(Base):
#     __tablename__ = 'app_genre'
#     genre_id = Column(Integer, ForeignKey('genre.genre_id'),
#                       primary_key=True)
#     steam_appid = Column(String(250), ForeignKey('dirty_data.steam_appid'),
#                          primary_key=True)

def open_db_connection():
    # specify database configurations
    config = settings.config
    db_user = settings.config.get('user')
    db_pwd = config.get('password')
    db_host = config.get('host')
    db_port = config.get('port')
    db_name = config.get('database')
    # specify connection string
    connection_str = f'mysql+pymysql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}'
    # connect to database
    engine = create_engine(connection_str)
    connection = engine.connect()

    return engine, connection
#
#
# def create_new_tables():
#     conn = open_db_connection()
#
#     meta = db.MetaData()
#
#     steam_appids = db.Table(
#         'steam_appids', meta,
#         db.Column('appid', db.Integer, primary_key=True),
#         db.Column('last_update', db.DateTime)
#     )
#
#     raw_data = db.Table(
#         'raw_data', meta,
#         db.Column('appid', db.Integer, primary_key=True),
#         db.Column('success', db.Boolean),
#         db.Column('data', db.Text),
#     )
#
#     meta.create_all(conn)
#
# if __name__ == "main":
#     create_new_tables()
