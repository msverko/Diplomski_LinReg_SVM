CREATE TABLE public.rt_bool_alm
(
	bool_alm_id serial PRIMARY KEY,
	bool_alm_rec_num integer NOT NULL,
	bool_alm_operator_id smallint,
	bool_alm_tag_name varchar(80),
	bool_alm_tag_value smallint
);

CREATE TABLE public.rt_bool_cmd
(
	bool_cmd_id serial PRIMARY KEY,
	bool_cmd_rec_num integer NOT NULL,
	bool_cmd_operator_id smallint,
	bool_cmd_tag_name varchar(80),
	bool_cmd_tag_value smallint
);

CREATE TABLE public.rt_bool_stat
(
	bool_stat_id serial PRIMARY KEY,
	bool_stat_rec_num integer NOT NULL,
	bool_stat_operator_id smallint,
	bool_stat_tag_name varchar(80),
	bool_stat_tag_value smallint
);

CREATE TABLE public.rt_int_set
(
	int_set_id serial PRIMARY KEY,
	int_set_rec_num integer NOT NULL,
	int_set_operator_id smallint,
	int_set_tag_name varchar(80),
	int_set_tag_value integer
);

CREATE TABLE public.rt_int_stat
(
	int_stat_id serial PRIMARY KEY,
	int_stat_rec_num integer NOT NULL,
	int_stat_operator_id smallint,
	int_stat_tag_name varchar(80),
	int_stat_tag_value integer
);

CREATE TABLE public.rt_real_stat
(
	real_stat_id serial PRIMARY KEY,
	real_stat_rec_num integer NOT NULL,
	real_stat_operator_id smallint,
	real_stat_tag_name varchar(80),
	real_stat_tag_value real
);



