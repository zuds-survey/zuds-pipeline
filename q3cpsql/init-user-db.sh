#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
CREATE USER ztfcoadd_admin;
CREATE DATABASE ztfcoadd;
GRANT ALL PRIVILEGES ON DATABASE ztfcoadd TO ztfcoadd_admin;
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "ztfcoadd" <<-EOSQL
CREATE EXTENSION q3c;
EOSQL

psql -v ON_ERROR_STOP=1 --username "ztfcoadd_admin" --dbname "ztfcoadd" -f /docker-entrypoint-initdb.d/createdb.psql