USE [master]
GO
/****** Object:  Database [Inventory_APP_DB]    Script Date: 20/04/2024 20:23:40 ******/
CREATE DATABASE [Inventory_APP_DB]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'Inventory_APP_DB', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL\DATA\Inventory_APP_DB.mdf' , SIZE = 8192KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'Inventory_APP_DB_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL\DATA\Inventory_APP_DB_log.ldf' , SIZE = 8192KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [Inventory_APP_DB] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [Inventory_APP_DB].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [Inventory_APP_DB] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET ARITHABORT OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [Inventory_APP_DB] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [Inventory_APP_DB] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET  DISABLE_BROKER 
GO
ALTER DATABASE [Inventory_APP_DB] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [Inventory_APP_DB] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET RECOVERY FULL 
GO
ALTER DATABASE [Inventory_APP_DB] SET  MULTI_USER 
GO
ALTER DATABASE [Inventory_APP_DB] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [Inventory_APP_DB] SET DB_CHAINING OFF 
GO
ALTER DATABASE [Inventory_APP_DB] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [Inventory_APP_DB] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [Inventory_APP_DB] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [Inventory_APP_DB] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
EXEC sys.sp_db_vardecimal_storage_format N'Inventory_APP_DB', N'ON'
GO
ALTER DATABASE [Inventory_APP_DB] SET QUERY_STORE = ON
GO
ALTER DATABASE [Inventory_APP_DB] SET QUERY_STORE (OPERATION_MODE = READ_WRITE, CLEANUP_POLICY = (STALE_QUERY_THRESHOLD_DAYS = 30), DATA_FLUSH_INTERVAL_SECONDS = 900, INTERVAL_LENGTH_MINUTES = 60, MAX_STORAGE_SIZE_MB = 1000, QUERY_CAPTURE_MODE = AUTO, SIZE_BASED_CLEANUP_MODE = AUTO, MAX_PLANS_PER_QUERY = 200, WAIT_STATS_CAPTURE_MODE = ON)
GO
USE [Inventory_APP_DB]
GO
/****** Object:  Table [dbo].[alembic_version]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[alembic_version](
	[version_num] [varchar](32) NOT NULL,
 CONSTRAINT [alembic_version_pkc] PRIMARY KEY CLUSTERED 
(
	[version_num] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[date_table]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[date_table](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[date] [date] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
UNIQUE NONCLUSTERED 
(
	[date] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[forecast]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[forecast](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[material_id] [int] NOT NULL,
	[quantity] [float] NOT NULL,
	[forecasted_for_date] [date] NOT NULL,
	[timestamp] [datetime] NOT NULL,
	[date_id] [int] NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[inventory]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[inventory](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[material_id] [int] NOT NULL,
	[storage_location_id] [int] NOT NULL,
	[quantity] [float] NOT NULL,
	[timestamp] [datetime] NOT NULL,
	[date_id] [int] NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[location]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[location](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[name] [varchar](100) NOT NULL,
	[region_id] [int] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[material]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[material](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[material_number] [varchar](50) NOT NULL,
	[category] [varchar](50) NULL,
	[subcategory] [varchar](50) NULL,
	[material_type] [varchar](50) NULL,
	[description] [varchar](max) NULL,
	[status] [int] NULL,
	[jpdm_number] [varchar](50) NULL,
	[sap_print_number] [varchar](50) NULL,
	[mw_code] [varchar](50) NULL,
	[wst_code] [varchar](50) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
UNIQUE NONCLUSTERED 
(
	[material_number] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[material_price]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[material_price](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[material_id] [int] NOT NULL,
	[sourcer] [varchar](50) NOT NULL,
	[price] [float] NOT NULL,
	[effective_date] [datetime] NOT NULL,
	[timestamp] [datetime] NOT NULL,
	[date_id] [int] NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[material_system]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[material_system](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[material_id] [int] NOT NULL,
	[system_id] [int] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[open_order]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[open_order](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[material_id] [int] NOT NULL,
	[quantity] [float] NOT NULL,
	[open_order_date] [date] NOT NULL,
	[timestamp] [datetime] NOT NULL,
	[date_id] [int] NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[region]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[region](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[name] [varchar](100) NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY],
UNIQUE NONCLUSTERED 
(
	[name] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[sale]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[sale](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[material_id] [int] NOT NULL,
	[quantity_sold] [float] NOT NULL,
	[sale_date] [date] NOT NULL,
	[timestamp] [datetime] NOT NULL,
	[date_id] [int] NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[storage_location]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[storage_location](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[name] [varchar](100) NOT NULL,
	[location_id] [int] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[system]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[system](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[name] [varchar](100) NOT NULL,
	[description] [varchar](max) NULL,
	[usage] [varchar](100) NULL,
	[status] [int] NOT NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[your_table_name]    Script Date: 20/04/2024 20:23:41 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[your_table_name](
	[material_number] [int] NULL,
	[sap_print_number] [varchar](max) NULL,
	[mw_code] [varchar](max) NULL,
	[wst_code] [varchar](max) NULL,
	[material_type] [varchar](max) NULL,
	[sourcer] [varchar](max) NULL,
	[jpdm_number] [varchar](max) NULL,
	[status] [float] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
ALTER TABLE [dbo].[forecast]  WITH CHECK ADD FOREIGN KEY([date_id])
REFERENCES [dbo].[date_table] ([id])
GO
ALTER TABLE [dbo].[forecast]  WITH CHECK ADD FOREIGN KEY([material_id])
REFERENCES [dbo].[material] ([id])
GO
ALTER TABLE [dbo].[inventory]  WITH CHECK ADD FOREIGN KEY([date_id])
REFERENCES [dbo].[date_table] ([id])
GO
ALTER TABLE [dbo].[inventory]  WITH CHECK ADD FOREIGN KEY([material_id])
REFERENCES [dbo].[material] ([id])
GO
ALTER TABLE [dbo].[inventory]  WITH CHECK ADD FOREIGN KEY([storage_location_id])
REFERENCES [dbo].[storage_location] ([id])
GO
ALTER TABLE [dbo].[location]  WITH CHECK ADD FOREIGN KEY([region_id])
REFERENCES [dbo].[region] ([id])
GO
ALTER TABLE [dbo].[material_price]  WITH CHECK ADD FOREIGN KEY([date_id])
REFERENCES [dbo].[date_table] ([id])
GO
ALTER TABLE [dbo].[material_price]  WITH CHECK ADD FOREIGN KEY([material_id])
REFERENCES [dbo].[material] ([id])
GO
ALTER TABLE [dbo].[material_system]  WITH CHECK ADD FOREIGN KEY([material_id])
REFERENCES [dbo].[material] ([id])
GO
ALTER TABLE [dbo].[material_system]  WITH CHECK ADD FOREIGN KEY([system_id])
REFERENCES [dbo].[system] ([id])
GO
ALTER TABLE [dbo].[open_order]  WITH CHECK ADD FOREIGN KEY([date_id])
REFERENCES [dbo].[date_table] ([id])
GO
ALTER TABLE [dbo].[open_order]  WITH CHECK ADD FOREIGN KEY([material_id])
REFERENCES [dbo].[material] ([id])
GO
ALTER TABLE [dbo].[sale]  WITH CHECK ADD FOREIGN KEY([date_id])
REFERENCES [dbo].[date_table] ([id])
GO
ALTER TABLE [dbo].[sale]  WITH CHECK ADD FOREIGN KEY([material_id])
REFERENCES [dbo].[material] ([id])
GO
ALTER TABLE [dbo].[storage_location]  WITH CHECK ADD FOREIGN KEY([location_id])
REFERENCES [dbo].[location] ([id])
GO
USE [master]
GO
ALTER DATABASE [Inventory_APP_DB] SET  READ_WRITE 
GO
