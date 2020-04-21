#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
CREATE USER admin;
CREATE DATABASE zuds;
GRANT ALL PRIVILEGES ON DATABASE zuds TO admin;
ALTER ROLE admin WITH SUPERUSER;
EOSQL

psql -v ON_ERROR_STOP=1 --username "admin" --dbname "zuds" <<-EOSQL
CREATE EXTENSION q3c;
EOSQL

