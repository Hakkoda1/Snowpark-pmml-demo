import com.snowflake.snowpark._
import com.snowflake.snowpark.functions._
import com.snowflake.snowpark.types._
import org.pmml4s.model.Model

// Small class for creating model from pmml file
// params - modelPath : string
// returns - pmml4s.model.Model obj type

class XGBPMML(modelPath: String) {
 def getModel(): Model = {
   val xgb: Model = Model.fromFile( s"$modelPath/lib/XGB_titanic.pmml")

   return xgb
 }
}

// Main function
object Main {
  def main(args: Array[String]): Unit = {

    // Snowflake config options
    // Replace <__>
    val configs = Map (
      "URL" -> "<ACCOUNT_URL>",
      "USER" -> "<SNOWFLAKE_USERNAME",
      "PASSWORD" -> "<SNOWFLAKE_PASSWORD>",
      "ROLE" -> "<SNOWFLAKE_ROLE>",
      "WAREHOUSE" -> "<SNOWFLAKE_WH>",
      "DB" -> "<SNOWFLAKE_DB>",
      "SCHEMA" -> "<SNOWFLAKE_SCHEMA>"
    )

    // Build Session using configs Map for Snowflake connection
    val session = Session.builder.configs(configs).create


    // Build Struct based on Snowflake table used for dataframe
    val schema = StructType(
      StructField("PCLASS", LongType, nullable = true) ::
      StructField("AGE", DoubleType, nullable = true) ::
      StructField("SIBSP", LongType, nullable = true) ::
      StructField("PARCH", LongType, nullable = true) ::
      StructField("FARE", DoubleType, nullable = true) ::
      StructField("SEX_FEMALE", LongType, nullable = true) ::
      StructField("SEX_MALE", LongType, nullable = true) ::
      StructField("EMBARKED_C", LongType, nullable = true) ::
      StructField("EMBARKED_Q", LongType, nullable = true) ::
      StructField("EMBARKED_S", LongType, nullable = true) ::
      Nil
    )

    // Read data from Snowflake table and write to df using the schema struct created
    val titanic_Df = session.read.schema(schema).table("TITANIC_TABLE")

    // print df to ensure read was successful
    // titanic_Df.collect.foreach(println)

    // Declare str with file paths and upload files as dependencies to Snowflake
    val libPath = new java.io.File("").getAbsolutePath
    println(libPath)
    session.addDependency(s"$libPath/lib/pmml4s_2.12-0.9.11.jar")
    session.addDependency(s"$libPath/lib/spray-json_2.12-1.3.5.jar")
    session.addDependency(s"$libPath/lib/XGB_titanic.pmml")

    // Use XGBPMML class to build model object from pmml file from the path declared
    val model = new XGBPMML(libPath).getModel()

    // Create User Defined Functin object as transformationUDF with the params and data types used in the
    // Snowflake table and df
    val transformationUDF = udf((pclass: Long, age: Double, sibsp: Long, parch: Long,
    fare: Double, sex_female: Long, embarked_c: Long, embarked_q: Long, embarked_s: Long,
    sex_male: Long) => {
      val v = Array[Any](pclass, age, sibsp, parch, fare, embarked_c, embarked_q, embarked_s, sex_female, sex_male)
      model.predict(v).last.asInstanceOf[Long]
    })

    // Add transformationUDF results as a column to the df
    titanic_Df.withColumn("Survived", transformationUDF(
                                                    col("PCLASS"), col("AGE"),
                                                    col("SIBSP"), col("PARCH"),
                                                    col("FARE"),  col("SEX_FEMALE"),
                                                    col("SEX_MALE"), col("EMBARKED_C"),
                                                    col("EMBARKED_Q"), col("EMBARKED_S")
                                                  ))
                                                  
                                                  
    // Print df
    titanic_Df.show()

  }
}