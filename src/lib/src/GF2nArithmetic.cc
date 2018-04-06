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

#include "../include/GF2nArithmeticOpenSSL.h"
#include "../include/GF2nArithmeticCuda.h"

namespace libcumffa {

	/**************************************************************************\

						class GF2nArithmetic implementations

	\**************************************************************************/
	
	GF2nArithmetic::GF2nArithmetic( std::string mode, GF2nArithmeticInterface *element )
	{
		m_mode = mode;
		m_element.reset(element);
	}

	GF2nArithmetic::~GF2nArithmetic() {}

	void GF2nArithmetic::setFieldSize( const uint32 field_size )
	{
		m_element->setFieldSize(field_size);
	}

	void GF2nArithmetic::setDummyParameters( const uint32 field_size, const std::string irred_poly )
	{
		m_element->setDummyParameters(field_size, irred_poly);
	}

	void GF2nArithmetic::setDummyParameters( const uint32 field_size, const unsigned char *irred_poly, const uint32 chunks_irred_poly )
	{
		m_element->setDummyParameters(field_size, irred_poly, chunks_irred_poly);
	}

	void GF2nArithmetic::setDummyParameters( const uint32 field_size, const void *irred_poly, const uint32 chunks_irred_poly )
	{
		m_element->setDummyParameters(field_size, irred_poly, chunks_irred_poly);
	}

	void GF2nArithmetic::setFlags( const unsigned char flags )
	{
		m_element->setFlags(flags);
	}

	GF2nArithmeticElement GF2nArithmetic::getElement( const std::string value )
	{
		GF2nArithmeticElement res = m_element->getElement(value);
		return res;
	}

	GF2nArithmeticElement GF2nArithmetic::getElement( const unsigned char *value, const uint32 chunks_value )
	{
		GF2nArithmeticElement res = m_element->getElement(value, chunks_value);
		return res;
	}

	GF2nArithmeticElement GF2nArithmetic::getElement( const void *value, const uint32 chunks_value )
	{
		GF2nArithmeticElement res = m_element->getElement(value, chunks_value);
		return res;
	}

	std::string GF2nArithmetic::getMode()
	{
		return m_mode;
	}


	/**************************************************************************\

					class GF2nArithmeticElementNull implementations

	\**************************************************************************/

	GF2nArithmeticElementNull::~GF2nArithmeticElementNull()
	{
	}

	GF2nArithmeticElementInterface *GF2nArithmeticElementNull::add( GF2nArithmeticElementInterface *other )
	{	
		return new GF2nArithmeticElementNull();
	}

	GF2nArithmeticElementInterface *GF2nArithmeticElementNull::sub( GF2nArithmeticElementInterface *other )
	{
		return new GF2nArithmeticElementNull();
	}		

	GF2nArithmeticElementInterface *GF2nArithmeticElementNull::mul( GF2nArithmeticElementInterface *other )
	{
		return new GF2nArithmeticElementNull();		
	}

	GF2nArithmeticElementInterface *GF2nArithmeticElementNull::div( GF2nArithmeticElementInterface *other )
	{
		return new GF2nArithmeticElementNull();
	}		

	GF2nArithmeticElementInterface *GF2nArithmeticElementNull::runWithElement( const std::string &what, GF2nArithmeticElementInterface *other )
	{
		return new GF2nArithmeticElementNull();
	}

	GF2nArithmeticElementInterface *GF2nArithmeticElementNull::runWithValue( const std::string &what, uint32 value )
	{
		return new GF2nArithmeticElementNull();
	}

	std::string GF2nArithmeticElementNull::toString()
	{
		std::string ret;	
		return ret;
	}

	void GF2nArithmeticElementNull::getValue( std::vector<uint8_t> &value )
	{
	}

	std::string GF2nArithmeticElementNull::getMetrics()
	{
		std::string dummy;
		return dummy;	
	}

	std::string GF2nArithmeticElementNull::getMetrics( const std::string &metrics_name )
	{
		std::string dummy;
		return dummy;
	}

	void GF2nArithmeticElementNull::setProperty( const std::string &property_name, const std::string &property_value )
	{
	}



	/**************************************************************************\

						class GF2nArithmeticElement implementations

	\**************************************************************************/
	
	GF2nArithmeticElement::GF2nArithmeticElement()
	{
		m_element.reset(new GF2nArithmeticElementNull());
	}

	GF2nArithmeticElement::GF2nArithmeticElement( GF2nArithmeticElementInterface *element )
	{
		m_element.reset(element);
	}

	void GF2nArithmeticElement::operator=( GF2nArithmeticElementInterface *element )
	{
		m_element.reset(element);
	}

	GF2nArithmeticElement::~GF2nArithmeticElement() {}

	const GF2nArithmeticElement operator+( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs )
	{
		GF2nArithmeticElement res = GF2nArithmeticElement(lhs.m_element.get()->add(rhs.m_element.get()));
		return res;
	}

	const GF2nArithmeticElement operator-( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs )
	{
		GF2nArithmeticElement res = GF2nArithmeticElement(lhs.m_element.get()->sub(rhs.m_element.get()));
		return res;
	}

	const GF2nArithmeticElement operator*( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs )
	{
		GF2nArithmeticElement res = GF2nArithmeticElement(lhs.m_element.get()->mul(rhs.m_element.get()));
		return res;
	}

	const GF2nArithmeticElement operator/( GF2nArithmeticElement const& lhs, GF2nArithmeticElement const& rhs )
	{
		GF2nArithmeticElement res = GF2nArithmeticElement(lhs.m_element.get()->div(rhs.m_element.get()));
		return res;
	}

	std::ostream& operator<<( std::ostream &out, GF2nArithmeticElement &elem )
	{
		out << elem.toString();
		return out;
	}

	const GF2nArithmeticElement GF2nArithmeticElement::runWithElement( const std::string &what, GF2nArithmeticElement const& other )
	{
		GF2nArithmeticElement res = GF2nArithmeticElement(m_element->runWithElement(what, other.m_element.get()));
		return res;
	}

	const GF2nArithmeticElement GF2nArithmeticElement::runWithValue( const std::string &what, uint32 value )
	{
		GF2nArithmeticElement res = GF2nArithmeticElement(m_element->runWithValue(what, value));
		return res;
	}	

	std::string GF2nArithmeticElement::toString()
	{
		return m_element->toString();
	}

	void GF2nArithmeticElement::getValue( std::vector<uint8_t> &value )
	{
		m_element->getValue(value);
	}

	std::string GF2nArithmeticElement::getMetrics()
	{
		return m_element->getMetrics();	
	}

	std::string GF2nArithmeticElement::getMetrics( const std::string &metrics_name )
	{
		return m_element->getMetrics(metrics_name);
	}

	void GF2nArithmeticElement::setProperty( const std::string &property_name, const std::string &property_value )
	{
		return m_element->setProperty(property_name, property_value);
	}

	/**************************************************************************\

						class GF2nArithmeticFactory implementations

	\**************************************************************************/
	
	GF2nArithmetic GF2nArithmeticFactory::createInstance( const std::string mode, const uint32 field_size )
	{
	 	if( mode.compare("OpenSSL") == 0 )
		{
			GF2nArithmetic obj = GF2nArithmetic(mode, new cpu::GF2nArithmeticOpenSSL());
			if( field_size != 0 )
				obj.setFieldSize(field_size);
			return obj;
		}
		else if( mode.compare("Cuda") == 0 )
		{
			GF2nArithmetic obj = GF2nArithmetic(mode, new gpu::GF2nArithmeticCuda());
			if( field_size != 0 )
				obj.setFieldSize(field_size);
			return obj;
		}
		else
		{
			throw ModeNotFoundException(mode);
		}
	}

	GF2nArithmetic* GF2nArithmeticFactory::createInstancePtr( const std::string mode, const uint32 field_size )
	{
		GF2nArithmetic *obj = NULL;

	 	if( mode.compare("OpenSSL") == 0 )
		{
			obj = new GF2nArithmetic(mode, new cpu::GF2nArithmeticOpenSSL());
			if( field_size != 0 )
				obj->setFieldSize(field_size);
		}
		else if( mode.compare("Cuda") == 0 )
		{
			obj = new GF2nArithmetic(mode, new gpu::GF2nArithmeticCuda());
			if( field_size != 0 )
				obj->setFieldSize(field_size);
		}
		else
		{
			throw ModeNotFoundException(mode);
		}

		return obj;
	}	

}