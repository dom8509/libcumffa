/*
 * cubffa (CUda Binary Finite Field Arithmetic library) provides 
 * functions for large binary galois field arithmetic on GPUs. 
 * Besides CUDA it is also possible to extend cubffa to any other 
 * underlying framework.
 * Copyright (C) 2016  Dominik Stamm
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __GF2N_ARITHMETIC_H__
#define __GF2N_ARITHMETIC_H__

#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

#include "CumffaTypes.h"
#include "GF2nArithmeticUtils.h"

namespace libcumffa {

	class ModeNotFoundException : public std::exception
	{
	public:
		ModeNotFoundException( std::string mode ) 
		{
			std::stringstream ss;
			ss << "Mode" << mode << " does not exist!!!";
			m_str = ss.str();
		}

	private:
		virtual const char* what() const throw()
		{
			return m_str.c_str();
		}

	private:
		std::string m_str;
	};

	class GF2nArithmeticElement;

	///////////////////////////////////////////////////////////////////////
	/*
		the Interface for GF2nArithmetic Objects
	*/
	class GF2nArithmeticInterface
	{
	public:
		virtual ~GF2nArithmeticInterface() {}
		virtual void setFieldSize( const uint32 field_size ) = 0;
		virtual void setDummyParameters( const uint32 field_size, const std::string irred_poly ) = 0;
		virtual void setDummyParameters( const uint32 field_size, const unsigned char *irred_poly, const uint32 chunks_irred_poly ) = 0;
		virtual void setDummyParameters( const uint32 field_size, const void *irred_poly, const uint32 chunks_irred_poly ) = 0;
		virtual void setFlags( const unsigned char flags ) = 0;
		virtual GF2nArithmeticElement getElement( const std::string value ) = 0;
		virtual GF2nArithmeticElement getElement( const unsigned char *value, const uint32 chunks_value ) = 0;
		virtual GF2nArithmeticElement getElement( const void *value, const uint32 chunks_value ) = 0;
	};

	///////////////////////////////////////////////////////////////////////
	/*
		wrapper for GF2nArithmeticInterface Objects
	*/
	class GF2nArithmetic
	{
	public:
		GF2nArithmetic( std::string mode, GF2nArithmeticInterface *element );
		~GF2nArithmetic();

	public:
		void setFieldSize( const uint32 field_size );
		void setDummyParameters( const uint32 field_size, const std::string irred_poly );
		void setDummyParameters( const uint32 field_size, const unsigned char *irred_poly, const uint32 chunks_irred_poly );
		void setDummyParameters( const uint32 field_size, const void *irred_poly, const uint32 chunks_irred_poly );
		void setFlags( const unsigned char flags );
		GF2nArithmeticElement getElement( const std::string value );
		GF2nArithmeticElement getElement( const unsigned char *value, const uint32 chunks_value );
		GF2nArithmeticElement getElement( const void *value, const uint32 chunks_value );
		std::string getMode();

	private:
		std::shared_ptr<GF2nArithmeticInterface> m_element;
		std::string m_mode;
	};

	///////////////////////////////////////////////////////////////////////
	/*
		the Interface for GF2nArithmetic Elements
	*/
	class GF2nArithmeticElementInterface
	{
	public:
		virtual ~GF2nArithmeticElementInterface() {}
		virtual GF2nArithmeticElementInterface *add( GF2nArithmeticElementInterface *other ) = 0;
		virtual GF2nArithmeticElementInterface *sub( GF2nArithmeticElementInterface *other ) = 0;
		virtual GF2nArithmeticElementInterface *mul( GF2nArithmeticElementInterface *other ) = 0;
		virtual GF2nArithmeticElementInterface *div( GF2nArithmeticElementInterface *other ) = 0;
		virtual GF2nArithmeticElementInterface *runWithElement( const std::string &what, GF2nArithmeticElementInterface *other ) = 0;
		virtual GF2nArithmeticElementInterface *runWithValue( const std::string &what, uint32 value ) = 0;
		virtual std::string toString() = 0;
		virtual void getValue( std::vector<uint8_t> &value ) = 0;
		virtual std::string getMetrics() = 0;
		virtual std::string getMetrics( const std::string &metrics_name ) = 0;
		virtual void setProperty( const std::string &property_name, const std::string &property_value ) = 0;
	};

	class GF2nArithmeticElementNull : public GF2nArithmeticElementInterface
	{
	public:
		virtual ~GF2nArithmeticElementNull();
		virtual GF2nArithmeticElementInterface *add( GF2nArithmeticElementInterface *other );
		virtual GF2nArithmeticElementInterface *sub( GF2nArithmeticElementInterface *other );
		virtual GF2nArithmeticElementInterface *mul( GF2nArithmeticElementInterface *other );
		virtual GF2nArithmeticElementInterface *div( GF2nArithmeticElementInterface *other );
		virtual GF2nArithmeticElementInterface *runWithElement( const std::string &what, GF2nArithmeticElementInterface *other );
		virtual GF2nArithmeticElementInterface *runWithValue( const std::string &what, uint32 value );
		virtual std::string toString();
		virtual void getValue( std::vector<uint8_t> &value );
		virtual std::string getMetrics();
		virtual std::string getMetrics( const std::string &metrics_name );
		virtual void setProperty( const std::string &property_name, const std::string &property_value );
	};	

	///////////////////////////////////////////////////////////////////////
	/*
		the wrapper for GF2nArithmeticElement that uses objects of the
		kind GF2nArithmeticElementInterface
	*/
	class GF2nArithmeticElement
	{
	public:
		GF2nArithmeticElement();
		GF2nArithmeticElement( GF2nArithmeticElementInterface *element );
		void operator=( GF2nArithmeticElementInterface *element );
		~GF2nArithmeticElement();

	public:
		friend const GF2nArithmeticElement operator+( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs );
		friend const GF2nArithmeticElement operator-( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs );
		friend const GF2nArithmeticElement operator*( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs );
		friend const GF2nArithmeticElement operator/( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs );
		friend std::ostream& operator<<( std::ostream &out, GF2nArithmeticElement &elem );
		const GF2nArithmeticElement runWithElement( const std::string &what, GF2nArithmeticElement const& other );
		const GF2nArithmeticElement runWithValue( const std::string &what, uint32 value );
		std::string toString();
		void getValue( std::vector<uint8_t> &value );
		std::string getMetrics();
		std::string getMetrics( const std::string &metrics_name );
		void setProperty( const std::string &property_name, const std::string &property_value );

	private:
		std::shared_ptr<GF2nArithmeticElementInterface> m_element;
	};


	const GF2nArithmeticElement operator+( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs );
	const GF2nArithmeticElement operator-( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs );
	const GF2nArithmeticElement operator*( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs );
	const GF2nArithmeticElement operator/( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs );
	std::ostream& operator<<( std::ostream &out, GF2nArithmeticElement &elem );

	///////////////////////////////////////////////////////////////////////
	/*
		the factory that creates a GF2nArithmetic
	*/
	class GF2nArithmeticFactory
	{
	private:
		GF2nArithmeticFactory() {}
	public:
		static GF2nArithmetic createInstance( const std::string mode, const uint32 field_size=0 );
		static GF2nArithmetic* createInstancePtr( const std::string mode, const uint32 field_size=0 );
	};		

	///////////////////////////////////////////////////////////////////////
	/*
		helper functions
	*/
	namespace utils {
		void convertStringToArray( std::string str, int size_chunk_bits, int bn_num_chunks, std::vector<std::string> &arr );
		void convertStringToArray2( std::string str, int size_chunk_bits, int bn_num_chunks, std::vector<std::string> &arr );
		void convertArrayToString( std::vector<std::string> &arr, int size_chunk_bits, int num_chunks, std::string &str );
	}
}

#endif //__GF2N_ARITHMETIC_H__